# auth.py — Auth "maison" très simple (admin/admin par défaut via .env)
import os
import streamlit as st

def _login_form():
    st.markdown("## 🔐 Connexion")
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Nom d'utilisateur", value="", key="__login_user__")
        p = st.text_input("Mot de passe", type="password", value="", key="__login_pass__")
        ok = st.form_submit_button("Se connecter")
    return ok, u.strip(), p

def ensure_auth():
    """
    Renvoie True si l’utilisateur est authentifié, sinon rend le formulaire
    et renvoie False. Utilise uniquement l’état Streamlit (pas de cookies).
    """
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
        st.session_state.auth_user = None

    # Désactiver l’auth en mettant STREAMLIT_AUTH=OFF dans le .env
    AUTH_ON = (os.getenv("STREAMLIT_AUTH", "ON").upper() in ("1", "TRUE", "ON", "YES"))
    if not AUTH_ON:
        st.session_state.auth_ok = True
        st.session_state.auth_user = "public"
        return True

    if st.session_state.auth_ok:
        return True

    expected_user = os.getenv("AUTH_DEFAULT_USER", "admin")
    expected_pwd  = os.getenv("AUTH_DEFAULT_PASSWORD", "admin")

    ok, u, p = _login_form()
    if ok:
        if u == expected_user and p == expected_pwd:
            st.session_state.auth_ok = True
            st.session_state.auth_user = u
            st.rerun()
        else:
            st.error("Identifiants invalides.")
    return False

def render_logout():
    if st.sidebar.button("Se déconnecter"):
        st.session_state.clear()
        st.rerun()
