import reflex as rx

config = rx.Config(
    app_name="streamlit_ai_agents",
    db_url="sqlite:///reflex.db",
    env=rx.Env.DEV,
)
