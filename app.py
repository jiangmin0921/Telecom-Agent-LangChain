# app.py

from ui.gradio_app import build_ui

if __name__ == "__main__":
    # 构建并启动Gradio应用
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)

