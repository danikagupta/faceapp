import streamlit as st
from streamlit.logger import get_logger
from streamlit_extras.switch_page_button import switch_page

LOGGER = get_logger(__name__)

# def switch_page(page_name: str):
#     from streamlit import _RerunData, _RerunException
#     from streamlit.source_util import get_pages

#     def standardize_name(name: str) -> str:
#         return name.lower().replace("_", " ")
    
#     page_name = standardize_name(page_name)

#     pages = get_pages("Home.py")  # OR whatever your main page is called

#     for page_hash, config in pages.items():
#         if standardize_name(config["page_name"]) == page_name:
#             raise _RerunException(
#                 _RerunData(
#                     page_script_hash=page_hash,
#                     page_name=page_name,
#                 )
#             )

#     page_names = [standardize_name(config["page_name"]) for config in pages.values()]

#     raise ValueError(f"Could not find page {page_name}. Must be one of {page_names}")


def run():
    st.set_page_config(
        page_title="KidSafe",
        page_icon="🧒",
    )

    st.write("# Welcome to KidSafe 🧒")

    (col1,col2,col3)=st.columns(3)
    with col1:
        if st.button("Find Your Kid Online"):
            switch_page("Find_Your_Kid_Online")
    with col2:
        if st.button("Learn More"):
            switch_page("Learn_More")
    with col3:
        if st.button("Contribute to our Safety Community"):
            switch_page("Contribute_to_our_Safety_Community")




if __name__ == "__main__":
    run()
