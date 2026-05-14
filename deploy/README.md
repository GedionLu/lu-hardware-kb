# deploy/ — 部署配置

## 文件清单

### chatbot_server.service
systemd 服务配置，用于生产环境部署。

## 部署步骤

```bash
# 安装服务
sudo cp deploy/chatbot_server.service /etc/systemd/system/

# 启动 + 开机自启
sudo systemctl daemon-reload
sudo systemctl enable --now chatbot_server

# 查看状态
sudo systemctl status chatbot_server

# 查看日志
sudo journalctl -u chatbot_server -f
```

## 注意事项

- 服务以 `admin` 用户运行
- 工作目录需调整为实际路径
- Qdrant 和 embed_server 需先启动
