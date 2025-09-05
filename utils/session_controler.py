import streamlit as st


def initialize(variable: str, value = None):
    if variable not in st.session_state:
        st.session_state[variable] = value

def set(variable: str, value):
    st.session_state[variable] = value

def get(variable: str):
    return st.session_state[variable]

def gets(variable: str):
    """Get variable if exists, else return None."""
    if variable in st.session_state:
        return st.session_state[variable]
    else:
        return None

def append(variable: str, value, unique: bool = False):
    if st.session_state[variable] is None:
        st.session_state[variable] = []
    if variable not in st.session_state:
        st.session_state[variable] = value
    else:
        if isinstance(value, (list, tuple)):
            for val in value:
                if not unique or val not in st.session_state[variable]:
                    st.session_state[variable].append(val)
        else:
            if not unique or value not in st.session_state[variable]:
                st.session_state[variable].append(value)
            
def remove(variable: str, value):
    if variable not in st.session_state:
        return
    else:
        if type(st.session_state[variable]) == list:
            st.session_state[variable].remove(value)

def delete(variable: str):
    if variable in st.session_state:
        del st.session_state[variable]

def check_exists(variable: str):
    return variable in st.session_state

def check_empty(variable: str):
    return variable not in st.session_state or st.session_state[variable] is None or st.session_state[variable] == "" or st.session_state[variable] == [] or st.session_state[variable] == {}


