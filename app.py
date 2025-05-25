import streamlit as st


saham =st.Page(
    "menu/saham.py", title="stock", icon=":material/dashboard:"
)
price =st.Page(
    "menu/price.py", title="price", icon=":material/dashboard:"
)


pg=st.navigation(
    {
        "menu" : [saham,price]
    }
)

pg.run()