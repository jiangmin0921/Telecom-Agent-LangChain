# app.py

import logging
import sys
from ui.gradio_app import build_ui

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """主函数：构建并启动 Gradio 应用"""
    try:
        logger.info("正在初始化电信行业智能对话系统...")
        demo = build_ui()
        logger.info("Gradio UI 构建成功，正在启动服务器...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,  # 设置为 True 可生成公网链接
            show_error=True
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"应用启动失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

