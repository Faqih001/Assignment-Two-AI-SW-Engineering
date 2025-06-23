import streamlit as st
from users import UserManager

def show_login_page():
    st.title("AI Tools Demo - Login")
    
    # Initialize user manager
    user_manager = UserManager()
    
    # Create tabs for Login and Sign Up
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    # Login Tab
    with tab1:
        st.header("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if user_manager.verify_user(login_username, login_password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = login_username
                st.success("Successfully logged in!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
    
    # Sign Up Tab
    with tab2:
        st.header("Create Account")
        new_username = st.text_input("Username", key="new_username")
        new_password = st.text_input("Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm Password", type="password")
        email = st.text_input("Email")
        
        if st.button("Create Account"):
            if not new_username or not new_password or not email:
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                if user_manager.create_user(new_username, new_password, email):
                    st.success("Account created successfully! Please log in.")
                    # Clear the form
                    st.session_state['new_username'] = ''
                    st.session_state['new_password'] = ''
                else:
                    st.error("Username already exists")
