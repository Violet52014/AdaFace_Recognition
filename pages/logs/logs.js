const api = require('../../utils/api.js')

Page({
  data: {
    logs: [],
    page: 1,
    size: 20,
    hasMore: true,
    loading: false,
    totalCount: 0,
    successCount: 0,
    errorCount: 0
  },

  onLoad() {
    this.loadRecords(true)
  },

  onShow() {
    // 每次显示页面时刷新数据
    this.loadRecords(true)
  },

  loadRecords(reset = false) {
    if (this.data.loading) return
    if (reset) {
      this.setData({ page: 1, logs: [], hasMore: true })
    }
    
    if (!this.data.hasMore) return
    
    this.setData({ loading: true })
    
    api.getRecognitionRecords({
      page: this.data.page,
      size: this.data.size
    })
      .then(res => {
        const newLogs = reset ? res.records : [...this.data.logs, ...res.records]
        
        // 计算统计数据
        let successCount = 0
        let errorCount = 0
        newLogs.forEach(log => {
          if (log.status === 'success') {
            successCount++
          } else {
            errorCount++
          }
        })
        
        this.setData({
          logs: newLogs,
          page: this.data.page + 1,
          hasMore: res.hasMore,
          loading: false,
          totalCount: res.total,
          successCount: successCount,
          errorCount: errorCount
        })
      })
      .catch(err => {
        wx.showToast({
          title: err.message || '加载失败',
          icon: 'none'
        })
        this.setData({ loading: false })
      })
  },

  loadMore() {
    this.loadRecords()
  },

  clearRecords() {
    wx.showModal({
      title: '确认清空',
      content: '确定要清空所有识别记录吗？',
      success: (res) => {
        if (res.confirm) {
          api.clearRecords()
            .then(() => {
              wx.showToast({
                title: '清空成功',
                icon: 'success'
              })
              this.loadRecords(true)
            })
            .catch(err => {
              wx.showToast({
                title: err.message || '清空失败',
                icon: 'none'
              })
            })
        }
      }
    })
  }
})