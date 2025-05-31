import streamlit as st


netflix_anlysis =st.Page(
    "project_sample/netflix.py", title="netflix analysis", icon=":material/dashboard:"
)
price =st.Page(
    "project_sample/price.py", title="price", icon=":material/dashboard:"
)


pg=st.navigation(
    {
        "Project Sample" : [netflix_anlysis,price]
    }
)

pg.run()