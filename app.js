App({
  onLaunch() {
    console.log('人脸识别门禁系统启动')
    
    // 检查摄像头权限
    wx.getSetting({
      success(res) {
        if (!res.authSetting['scope.camera']) {
          wx.authorize({
            scope: 'scope.camera',
            success() {
              console.log('摄像头权限已授权')
            }
          })
        }
      }
    })
  }
})