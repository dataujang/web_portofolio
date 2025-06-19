import streamlit as st


netflix_anlysis =st.Page(
    "project_sample/netflix.py", title="netflix analysis", icon=":material/dashboard:"
)
dinamis =st.Page(
    "project_sample/dinamis.py", title="dinamis", icon=":material/dashboard:"
)


pg=st.navigation(
    {
        "Project Sample" : [netflix_anlysis,dinamis]
    }
)

pg.run()