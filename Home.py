

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Home",
        page_icon="👋",
    )

    st.write("# KidSafe 👋")

    st.markdown(
        """
The average child has their picture shared on online 1,300 times before the age of 13 – that’s before they are even authorised to create their own social media profile on Facebook or Instagram [4].  In the age of generative AI - these images can be used for any number of harmful purposes and the damage can follow a child into adulthood [2,3].  While advanced users - like Mark Zuckerberg, take strong steps to avoid pictures of their kids online - many parents do not have the support needed to do the same [1].

Introducing KidSafe: A New Standard in Child Online Safety

KidSafe is not just another app; it's your child's online guardian. We've developed an innovative and comprehensive solution that focuses on three key pillars, ensuring your child's online safety without invading their privacy.

How does KidSafe work?
Image Embeddings & Similarity Search: KidSafe utilizes cutting-edge technology to find matching pictures without anyone storing your child's photos. We prioritize your child's privacy while offering a robust protective shield.
Steps
A parent uploads a child’s photo
An embedding is created that captures the essence of the picture into a vector
The vector is compared via similarity search to a database of images
The parent is given the URLs of all online images that appear to be that of their children. The parent can check each URL and determine if they are ok with the picture.
Community: KidSafe also includes a community feature where parents can engage to protect all children. Parents (or anyone else) can crowdsource URLs which will then be added to the search database. Images are never saved - only embeddings. 

Why Choose KidSafe?
Empowerment: KidSafe empowers parents to proactively ensure their child's online safety, while respecting privacy boundaries.
Visibility: KidSafe strikes a balance between trust and vigilance by providing insights into your child's online interactions, fostering open conversations about internet safety.
Connection: Strengthen the bond between parents and children through open discussions about online safety and making the internet a safer place for your kids.
Don't leave your child's online safety to chance. Choose KidSafe and take the first step in safeguarding your child's digital world. Your child's safety is our priority, and together, we can create a secure and enjoyable online journey.
Try KidSafe today. Your peace of mind begins here.

    """
    )


if __name__ == "__main__":
    run()
