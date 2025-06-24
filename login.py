import streamlit as st
from users import UserManager

def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "MNIST Classifier"
    if 'should_redirect' not in st.session_state:
        st.session_state['should_redirect'] = False

def show_login_page():
    """Display the login page and handle authentication"""
    init_session_state()
    st.title("AI Tools Demo - Login")
    
    # Initialize user manager
    user_manager = UserManager()
    
    # Create tabs for Login and Sign Up
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    # Login Tab
    with tab1:
        st.header("Login")
        with st.form("login_form", clear_on_submit=True):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login")
            
            if submit_login and login_username and login_password:
                if user_manager.verify_user(login_username, login_password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = login_username
                    st.success("Successfully logged in!")
                    st.session_state['should_redirect'] = True
                    st.rerun()
                    return True
                else:
                    st.error("Invalid username or password")
    
    # Sign Up Tab
    with tab2:
        st.header("Create Account")
        with st.form("signup_form", clear_on_submit=True):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            email = st.text_input("Email")
            submit_signup = st.form_submit_button("Create Account")
            
            if submit_signup:
                if not new_username or not new_password or not email:
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif user_manager.create_user(new_username, new_password, email):
                    st.success("Account created successfully! Please log in.")
                else:
                    st.error("Username already exists")
    
    if st.session_state.get('should_redirect', False):
        st.session_state['should_redirect'] = False
        st.rerun()
    
    return False
