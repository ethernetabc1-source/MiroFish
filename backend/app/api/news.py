"""
News API routes — Auto Source Collection
"""

import io
import re
import threading
import uuid

from flask import request, jsonify
from werkzeug.datastructures import FileStorage

from . import news_bp
from ..models.project import ProjectManager
from ..models.task import TaskManager, TaskStatus
from ..services.news_collector import NewsCollectorService
from ..services.text_processor import TextProcessor
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.news')


# ------------------------------------------------------------------ #
#  POST /api/news/collect                                              #
# ------------------------------------------------------------------ #

@news_bp.route('/collect', methods=['POST'])
def collect_news():
    """Start async news collection task."""
    data = request.get_json(silent=True) or {}
    question = (data.get('prediction_question') or '').strip()
    max_sources = int(data.get('max_sources', 20))

    if not question:
        return jsonify({'success': False, 'error': 'prediction_question is required'}), 400

    if max_sources < 1 or max_sources > 50:
        max_sources = 20

    task_manager = TaskManager()
    task_id = task_manager.create_task('news_collect', metadata={'question': question})
    task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0,
                             message='Starting news collection...')

    def _run():
        tm = TaskManager()
        try:
            collector = NewsCollectorService()

            def _cb(progress, message):
                tm.update_task(task_id, progress=progress, message=message)

            result = collector.collect(question, max_sources=max_sources, task_callback=_cb)
            tm.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=100,
                message=f"Collected {len(result['articles'])} articles.",
                result=result,
            )
        except Exception as e:
            logger.error(f"News collection failed: {e}", exc_info=True)
            tm.update_task(
                task_id,
                status=TaskStatus.FAILED,
                message=str(e),
                error=str(e),
            )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({'success': True, 'data': {'task_id': task_id}})


# ------------------------------------------------------------------ #
#  GET /api/news/task/<task_id>                                        #
# ------------------------------------------------------------------ #

@news_bp.route('/task/<task_id>', methods=['GET'])
def get_news_task(task_id):
    """Poll collection task status."""
    task = TaskManager().get_task(task_id)
    if not task:
        return jsonify({'success': False, 'error': 'Task not found'}), 404

    return jsonify({
        'success': True,
        'data': {
            'task_id': task_id,
            'status': task.status.value,
            'progress': task.progress,
            'message': task.message,
            'result': task.result,
            'error': task.error,
        }
    })


# ------------------------------------------------------------------ #
#  POST /api/news/create-project                                       #
# ------------------------------------------------------------------ #

@news_bp.route('/create-project', methods=['POST'])
def create_project_from_articles():
    """Create a MiroFish project from selected articles."""
    data = request.get_json(silent=True) or {}
    question = (data.get('prediction_question') or '').strip()
    article_ids = data.get('article_ids', [])
    articles = data.get('articles', [])  # full article objects

    if not question:
        return jsonify({'success': False, 'error': 'prediction_question is required'}), 400
    if not articles:
        return jsonify({'success': False, 'error': 'articles list is required'}), 400

    # Filter to only selected IDs if provided
    if article_ids:
        id_set = set(article_ids)
        articles = [a for a in articles if a.get('id') in id_set]

    if not articles:
        return jsonify({'success': False, 'error': 'No matching articles found'}), 400

    try:
        # Create project
        project_name = f"Auto: {question[:60]}"
        project = ProjectManager.create_project(name=project_name)
        project.simulation_requirement = question

        combined_texts = []

        for art in articles:
            title = art.get('title', 'article')
            text = art.get('text') or art.get('snippet') or ''
            if not text:
                continue

            # Add metadata header to each article
            # Truncate body to avoid token overload when 20 full articles
            # are later sent to OntologyGenerator
            truncated_text = text[:3000] if len(text) > 3000 else text

            article_content = (
                f"Title: {title}\n"
                f"Source: {art.get('source', '')}\n"
                f"Published: {art.get('published', '')}\n"
                f"URL: {art.get('url', '')}\n\n"
                f"{truncated_text}"
            )

            # Save as individual file — append short UUID to avoid filename collisions
            filename = f"{_safe_filename(title)}_{str(uuid.uuid4())[:6]}.txt"
            file_bytes = article_content.encode('utf-8', errors='replace')
            file_storage = FileStorage(
                stream=io.BytesIO(file_bytes),
                filename=filename,
                name='files',
                content_type='text/plain',
            )
            file_info = ProjectManager.save_file_to_project(
                project.project_id, file_storage, filename
            )
            project.files.append({
                'filename': file_info['original_filename'],
                'size': file_info['size'],
            })
            combined_texts.append(article_content)

        if not combined_texts:
            return jsonify({'success': False, 'error': 'None of the selected articles had extractable text'}), 400

        # Save combined extracted text
        combined = '\n\n---\n\n'.join(combined_texts)
        combined = TextProcessor.preprocess_text(combined)
        ProjectManager.save_extracted_text(project.project_id, combined)
        project.total_text_length = len(combined)

        ProjectManager.save_project(project)

        logger.info(f"Created project {project.project_id} with {len(articles)} articles")

        return jsonify({
            'success': True,
            'data': {
                'project_id': project.project_id,
                'project_name': project_name,
                'article_count': len(articles),
                'total_text_length': len(combined),
            }
        })

    except Exception as e:
        logger.error(f"Project creation failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _safe_filename(title: str) -> str:
    """Convert article title to a safe filename."""
    safe = re.sub(r'[^\w\s-]', '', title)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:60] or 'article'
