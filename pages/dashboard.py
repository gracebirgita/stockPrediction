import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import time

# st.write("# Dashboard")
def dashboard():
    # st.title("Dashboard page")

    # # Konfigurasi Streamlit
    # # st.set_page_config(page_title="Stock Prediction", layout="centered")
    # st.markdown(
    #     '<h3 style="color: gold;">Select dataset for prediction</h3>',
    #     unsafe_allow_html=True,
    # )
    # Membaca query parameter untuk navigasi
   
    st.title("Welcome to the Stock Prediction Web")
    st.write("")
    # st.divider()

    st.image("https://images.unsplash.com/photo-1651341050677-24dba59ce0fd?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", caption="Empowering Your Investment Decisions", use_container_width=True)

    
    # Penjelasan manfaat aplikasi
    # st.markdown("""
    # ### 
    # - 📊 **Data-Driven Insights**: Make informed investment decisions.
    # - 🚀 **Growth Opportunities**: Identify potential high-growth stocks early.
    # - 🔍 **Risk Management**: Minimize losses by predicting market trends accurately.
    # - 💡 **Empowered Decisions**: Combine AI predict stocks.
                

    # """)

    st.write("")
    st.write("")

    # Tombol Explore

    # if st.button("🚀 Explore Stock Prediction"):
    #     switch_page("predict")
        # navigate_to("explore")
    # Halaman Welcome
    st.markdown(
        '<h3 style="color: gold;">Stock</h3><br></br>',
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown(
        """
        <div style="display: flex; gap: 40px; justify-content: center;">
            <div style="background: rgba(255,255,255,0.5); padding: 12px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <img src="https://mainsaham.id/wp-content/uploads/2023/02/img-BBCA.png" width="100">
            </div>
            <div style="background: rgba(255,255,255,0.5); padding: 12px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/BRI_2020.svg/2560px-BRI_2020.svg.png" width="100">
            </div>
            <div style="background: rgba(255,255,255,0.5); padding: 12px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Bank_Mandiri_logo_2016.svg/2560px-Bank_Mandiri_logo_2016.svg.png" width="100">
            </div>
        </div>
        <br></br>
        """,
        unsafe_allow_html=True
    )
    # logo_urls = [
    #     "https://mainsaham.id/wp-content/uploads/2023/02/img-BBCA.png",
    #     "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/BRI_2020.svg/2560px-BRI_2020.svg.png",
    #     "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Bank_Mandiri_logo_2016.svg/2560px-Bank_Mandiri_logo_2016.svg.png",
    # ]
    # cols = st.columns(3)
    # for col, url in zip(cols, logo_urls):
    #     col.image(url, width=100)

    st.write("")
    st.write("")


    st.markdown(
        '<h3 style="color: gold;">Benefit</h3><br></br>',
        unsafe_allow_html=True,
    )    
    st.write("")
    st.write("")

    # Membagi layout menjadi empat kolom untuk tiap poin
    for idx, (title, description, img_url) in enumerate([
        ("Data-Driven Insights", "Make informed investment decisions.", "https://www.nec.com.au/application/files/6615/8942/5303/insights-blog-data-data-driven-step-1-content-image-nec_mr.png"),
        ("Growth Opportunities", "Identify potential high-growth stocks early.", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcWkKHPaXScbHq1xB-wcH_jgFg4wA9Y5Yd1Q&s"),
        ("Risk Management", "Minimize losses by predicting market trends accurately.", "https://static.vecteezy.com/system/resources/thumbnails/028/035/349/small_2x/risk-management-3d-illustration-png.png"),
        ("Empowered Decisions", "Combine AI to predict stocks effectively.", "https://static.vecteezy.com/system/resources/thumbnails/010/916/534/small/credit-card-and-smartphone-with-stock-market-app-on-screen-making-payments-or-transactions-png.png"),
    ]):
        col1, col2 = st.columns([3, 3])

        if idx % 2 ==0:
            with col1:
                st.image(img_url, use_container_width=True)
            with col2:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.markdown(f"### {title}")
                st.write(description)
        else:
            with col1:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.markdown(f"### {title}")
                st.write(description)
            with col2:
                st.image(img_url, use_container_width=True)

        st.write("")
        st.write("")
        st.write("")
        time.sleep(0.8)

    # Tombol interaktif untuk melanjutkan ke halaman berikutnya
    st.markdown("---")

    # st.markdown("""
    # <hr style="border:1px solid gray;margin-top:30px;margin-bottom:20px;">
    # """, unsafe_allow_html=True)
   
dashboard()