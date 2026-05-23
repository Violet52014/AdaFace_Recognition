// API接口配置
//const BASE_URL = 'https://your-server.com/api' // 替换为实际后端地址http://192.168.216.131:5000
const BASE_URL = 'http://192.168.150.131:5000/api'

const parseResponseData = (raw) => {
  if (typeof raw === 'string') {
    try {
      return JSON.parse(raw)
    } catch (e) {
      return null
    }
  }
  return raw
}

// 请求封装（JSON）
const request = (url, method, data, options = {}) => {
  return new Promise((resolve, reject) => {
    const fullUrl = `${BASE_URL}${url}`
    const showLoading = options.showLoading !== false
    console.log('[API REQUEST]', method, fullUrl, data || {})
    if (showLoading) {
      wx.showLoading({ title: '加载中', mask: true })
    }
    wx.request({
      url: fullUrl,
      method: method,
      data: data,
      timeout: options.timeout || 8000,
      header: {
        'content-type': 'application/json'
      },
      success: (res) => {
        console.log('[API SUCCESS]', method, fullUrl, res.statusCode, res.data)
        if (res.statusCode === 200 && res.data.code === 0) {
          resolve(res.data.data)
        } else {
          const message = (res.data && res.data.message) || '服务响应异常'
          reject(res.data)
          wx.showToast({
            title: message,
            icon: 'none',
            duration: 2500
          })
        }
      },
      fail: (err) => {
        console.error('[API FAIL]', method, fullUrl, err)
        let message = '网络请求失败'
        const errMsg = (err && err.errMsg) || ''
        if (errMsg.includes('timeout')) {
          message = '请求超时，请检查网络或后端服务'
        } else if (errMsg.includes('url not in domain list')) {
          message = '请求域名未加入白名单'
        } else if (errMsg.includes('fail')) {
          message = '无法连接到后端服务'
        }
        wx.showToast({
          title: message,
          icon: 'none',
          duration: 3000
        })
        reject({
          ...err,
          message
        })
      },
      complete: () => {
        if (showLoading) {
          wx.hideLoading()
        }
      }
    })
  })
}

/**
 * multipart 上传临时文件（与 JSON+Base64 二选一，后端同一路由 /face/recognize）
 * @param {string} filePath 本地临时路径
 * @param {object} options showLoading, timeout
 */
const recognizeFaceUpload = (filePath, options = {}) => {
  return new Promise((resolve, reject) => {
    const fullUrl = `${BASE_URL}/face/recognize`
    const showLoading = options.showLoading !== false
    console.log('[API UPLOAD]', fullUrl, filePath)
    if (showLoading) {
      wx.showLoading({ title: '加载中', mask: true })
    }
    wx.uploadFile({
      url: fullUrl,
      filePath: filePath,
      name: 'image',
      timeout: options.timeout || 8000,
      success: (res) => {
        const body = parseResponseData(res.data)
        console.log('[API UPLOAD SUCCESS]', res.statusCode, body)
        if (res.statusCode === 200 && body && body.code === 0) {
          resolve(body.data)
        } else {
          const message = (body && body.message) || '服务响应异常'
          reject(body || { message })
          wx.showToast({
            title: message,
            icon: 'none',
            duration: 2500
          })
        }
      },
      fail: (err) => {
        console.error('[API UPLOAD FAIL]', fullUrl, err)
        let message = '网络请求失败'
        const errMsg = (err && err.errMsg) || ''
        if (errMsg.includes('timeout')) {
          message = '请求超时，请检查网络或后端服务'
        } else if (errMsg.includes('url not in domain list')) {
          message = '请求域名未加入白名单'
        } else if (errMsg.includes('fail')) {
          message = '无法连接到后端服务'
        }
        wx.showToast({
          title: message,
          icon: 'none',
          duration: 3000
        })
        reject({
          ...err,
          message
        })
      },
      complete: () => {
        if (showLoading) {
          wx.hideLoading()
        }
      }
    })
  })
}

// API接口定义
module.exports = {
  // 人脸识别（JSON + Base64，兼容旧调用）
  recognizeFace: (imageBase64, options = {}) => request('/face/recognize', 'POST', { image: imageBase64 }, options),

  // 人脸识别（multipart 上传临时文件，推荐）
  recognizeFaceUpload,

  // 获取识别记录列表（轮询/列表易并发，不弹 Loading，避免 showLoading/hideLoading 配对异常）
  getRecognitionRecords: (params) =>
    request('/records/list', 'GET', params, { showLoading: false, timeout: 15000 }),

  // 获取今日统计（同上；统计接口偶发较慢，单独放宽超时）
  getTodayStats: () => request('/stats/today', 'GET', undefined, { showLoading: false, timeout: 15000 }),

  // 清空识别记录
  clearRecords: () => request('/records/clear', 'POST')
}
