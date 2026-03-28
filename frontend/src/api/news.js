import service, { requestWithRetry } from './index'

/**
 * Start automatic news source collection
 * @param {String} question - The prediction question
 * @param {Number} maxSources - Maximum articles to collect (default 20)
 * @returns {Promise} { task_id }
 */
export function startNewsCollection(question, maxSources = 20) {
  return requestWithRetry(() =>
    service({
      url: '/api/news/collect',
      method: 'post',
      data: { prediction_question: question, max_sources: maxSources }
    })
  )
}

/**
 * Poll news collection task status
 * @param {String} taskId
 * @returns {Promise} { status, progress, message, result }
 */
export function getNewsTaskStatus(taskId) {
  return service({
    url: `/api/news/task/${taskId}`,
    method: 'get'
  })
}

/**
 * Create a MiroFish project from selected articles
 * @param {String} question - The prediction question (becomes simulation_requirement)
 * @param {Array} articles - Full article objects to include
 * @param {Array} articleIds - Optional subset of IDs to select
 * @returns {Promise} { project_id, article_count }
 */
export function createProjectFromArticles(question, articles, articleIds = null) {
  return requestWithRetry(() =>
    service({
      url: '/api/news/create-project',
      method: 'post',
      data: {
        prediction_question: question,
        articles,
        article_ids: articleIds
      }
    })
  )
}
