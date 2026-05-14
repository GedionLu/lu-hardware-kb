# static/ — Web 前端

## 文件清单

### index.html
单页面聊天界面。入口由 `chatbot_server.py` 的 `GET /` 返回。

**功能：**
- 快捷示例按钮（OH430 加回车、1900 USB 等）
- 聊天历史（localStorage 持久化，最多 50 条）
- 清空对话
- 图片光箱放大
- 响应式布局（移动端适配）

---

### style.css
全局样式，300+ 行。

**设计：**
- CSS 变量（primary 色系）
- 消息气泡动画（slideIn / fadeIn）
- 图片 hover 放大效果
- 移动端响应式 (`@media max-width:480px`)
- 自定义滚动条

---

### app.js
前端交互逻辑，200+ 行。

**核心函数：**
- `send()` — 发送请求 `POST /ask`，处理 rate limit / 网络错误
- `render()` — 渲染消息列表（文本 + 图片）
- `loadHistory()` / `saveHistory()` — localStorage 持久化
- `openLightbox()` — 图片放大查看
- `clearChat()` — 清空对话（含确认）

**依赖：** fetch API（原生）、localStorage（历史存储）
