const api = require('../../utils/api.js')

Page({
  data: {
    cameraReady: false,
    recognizing: false,
    autoRecognizing: false,
    showResult: false,
    showFaceFrame: true,
    cameraTip: '请将面部对准取景框',
    frameIntervalMs: 333, // 3 FPS
    statsIntervalMs: 2000,
    cooldownMs: 800,
    sameNameDebounceMs: 2000,
    lastRecognizeAt: 0,
    lastRecognizedName: '',
    lastRecognizedTime: 0,
    faceBoxes: [],
    result: {
      type: '',
      name: '',
      time: '',
      message: ''
    },
    stats: {
      total: 0,
      success: 0,
      rate: 0
    }
  },

  onLoad() {
    this.loadTodayStats()
  },

  onShow() {
    // 切回页面时重置识别中状态，避免按钮文案卡在“识别中...”
    this.setData({ recognizing: false })
  },

  onHide() {
    // 切页时停止自动识别与统计轮询，回到页面后由用户重新开始
    this.stopStatsPolling()
    this.stopAutoRecognize()
  },

  onUnload() {
    this.stopStatsPolling()
    this.stopAutoRecognize()
    if (this.resultTimer) {
      clearTimeout(this.resultTimer)
    }
  },

  onCameraStop() {
    this.setData({ cameraReady: true, recognizing: false })
  },

  onCameraError() {
    wx.showToast({
      title: '摄像头启动失败，请检查权限',
      icon: 'none',
      duration: 3000
    })
    this.setData({ cameraReady: false })
    this.stopAutoRecognize()
  },

  toggleAutoRecognize() {
    if (!this.data.cameraReady) return
    if (this.data.autoRecognizing) {
      this.stopAutoRecognize()
    } else {
      this.startAutoRecognize()
    }
  },

  startAutoRecognize() {
    if (this.autoTimer || !this.data.cameraReady) return
    this.setData({
      autoRecognizing: true,
      recognizing: false,
      cameraTip: '自动识别中（3 FPS）'
    })
    this.startStatsPolling()
    this.captureAndRecognize()
    this.autoTimer = setInterval(() => {
      this.captureAndRecognize()
    }, this.data.frameIntervalMs)
  },

  stopAutoRecognize() {
    if (this.autoTimer) {
      clearInterval(this.autoTimer)
      this.autoTimer = null
    }
    this.stopStatsPolling()
    this.loadTodayStats()
    this.setData({
      autoRecognizing: false,
      recognizing: false,
      cameraTip: '请将面部对准取景框',
      faceBoxes: []
    })
  },

  startStatsPolling() {
    if (this.statsTimer) return
    this.statsTimer = setInterval(() => {
      this.loadTodayStats()
    }, this.data.statsIntervalMs)
  },

  stopStatsPolling() {
    if (this.statsTimer) {
      clearInterval(this.statsTimer)
      this.statsTimer = null
    }
  },

  captureAndRecognize() {
    if (!this.data.autoRecognizing || !this.data.cameraReady || this.data.recognizing) return
    const now = Date.now()
    if (now - this.data.lastRecognizeAt < this.data.cooldownMs) return

    const cameraContext = wx.createCameraContext()
    this.setData({ recognizing: true })

    cameraContext.takePhoto({
      quality: 'normal',
      success: (res) => {
        api.recognizeFaceUpload(res.tempImagePath, { showLoading: false })
          .then(result => {
            this.setData({ lastRecognizeAt: Date.now() })
            this.updateFaceBoxes(result.results || [])
            this.handleRecognitionResult(result)
          })
          .catch(err => {
            this.setData({ lastRecognizeAt: Date.now() })
            this.setData({ faceBoxes: [] })
            this.showRecognitionError(err.message || '识别失败，请重试')
          })
      },
      fail: () => {
        this.setData({ lastRecognizeAt: Date.now() })
        this.setData({ faceBoxes: [] })
        this.showRecognitionError('拍照失败，请重试')
      }
    })
  },

  updateFaceBoxes(results) {
    const faceBoxes = (results || [])
      .filter(item => item && item.bbox)
      .map(item => {
        const { x, y, w, h } = item.bbox
        return {
          left: `${x * 100}%`,
          top: `${y * 100}%`,
          width: `${w * 100}%`,
          height: `${h * 100}%`,
          label: item.label || item.name || '未识别'
        }
      })
    const now = Date.now()
    const labelSignature = faceBoxes.map(item => item.label).join('|')
    if (
      labelSignature &&
      labelSignature === this.data.lastRecognizedName &&
      now - this.data.lastRecognizedTime < this.data.sameNameDebounceMs
    ) {
      return
    }
    this.setData({ faceBoxes })
  },

  shouldSkipSameNameRefresh(result) {
    if (!result || !result.recognized || !result.name) return false
    const now = Date.now()
    const isSameName = result.name === this.data.lastRecognizedName
    const inDebounceWindow = now - this.data.lastRecognizedTime < this.data.sameNameDebounceMs
    if (isSameName && inDebounceWindow) {
      return true
    }
    this.setData({
      lastRecognizedName: result.name,
      lastRecognizedTime: now
    })
    return false
  },

  handleRecognitionResult(result) {
    if (this.shouldSkipSameNameRefresh(result)) {
      this.setData({ recognizing: false })
      return
    }
    const now = new Date()
    const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`
    
    if (result.recognized) {
      this.setData({
        showResult: true,
        result: {
          type: 'success',
          name: result.name,
          time: timeStr,
          message: result.message || `识别成功，欢迎 ${result.name}`
        }
      })
      
      // 震动反馈
      wx.vibrateShort()
    } else {
      this.setData({
        showResult: true,
        result: {
          type: 'error',
          name: '未识别',
          time: timeStr,
          message: result.message || '未识别到本班同学，请重试'
        }
      })
    }
    
    // 3秒后隐藏结果
    if (this.resultTimer) {
      clearTimeout(this.resultTimer)
    }
    this.resultTimer = setTimeout(() => {
      this.setData({ showResult: false })
    }, 3000)
    
    this.setData({ recognizing: false })
  },

  showRecognitionError(message) {
    const now = new Date()
    const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`
    
    this.setData({
      recognizing: false,
      showResult: true,
      result: {
        type: 'error',
        name: '错误',
        time: timeStr,
        message: message
      }
    })
    
    if (this.resultTimer) {
      clearTimeout(this.resultTimer)
    }
    this.resultTimer = setTimeout(() => {
      this.setData({ showResult: false })
    }, 3000)
  },

  loadTodayStats() {
    api.getTodayStats()
      .then(stats => {
        this.setData({ stats })
      })
      .catch(err => {
        console.error('加载统计数据失败', err)
        // 如果接口失败，显示默认数据
        this.setData({
          stats: {
            total: 0,
            success: 0,
            rate: 0
          }
        })
      })
  }
})