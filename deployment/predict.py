# Import Libraries
import streamlit as st
import pandas as pd
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import time

# Load the tokenizer and model for sentiment analysis
model_dir = './saved_model/'  # Update this path if your model is saved elsewhere

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Function to perform sentiment analysis
def predict_sentiment(texts):
    # Tokenize and encode the texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    # Move inputs to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)
        confidences = torch.max(probabilities, dim=-1).values

    # Map predictions to labels
    label_map = {0: 'Negative', 1: 'Positive'}
    predicted_labels = [label_map[pred.item()] for pred in predicted_classes]
    confidences = confidences.cpu().numpy()

    return predicted_labels, confidences

st.write('')

def run():
    st.title("🏃 Customer Churn Prediction")
    st.markdown('---')

    st.sidebar.markdown("---")
    
    st.sidebar.write("### BERT Model")
    st.sidebar.write("This application uses a BERT model to analyze the sentiment of customer feedback. BERT is highly effective at understanding context and nuances in text, even for long feedback, despite being trained on shorter sentences in our dataset.")
    st.sidebar.image("bert.jpg", width=200)
    st.sidebar.write("BERT (Bidirectional Encoder Representations from Transformers) works by understanding the context of words in a sentence by looking at the words that come before and after them. This bidirectional approach allows BERT to capture the meaning of words more accurately.")
    
    st.sidebar.markdown('---')  # Line separating between models
    
    st.sidebar.write("### SVM Classifier")
    st.sidebar.write("After sentiment analysis, an SVM classifier is used to predict whether a customer is likely to churn based on the sentiment and other features provided.")
    st.sidebar.image("svm.png", width=200)
    st.sidebar.write("SVM (Support Vector Machine) works by finding the hyperplane that best separates the data into different classes. In this case, it uses the sentiment and other features to predict customer churn.")

    st.write("## ✍🏻📑 Input Data: ")
    st.write("")  # Add a blank line for spacing
    st.write("This prediction will be done by first analyzing the sentiment from the feedback provided using a BERT model. The sentiment, along with other features, will then be used to classify whether the customer is likely to churn or not churn using an SVM classifier.")

    # Sample data
    samples = {
        "Sample 1": {
            "customer_id": "CUST001",
            "tenure": 12,
            "contract": "two year",
            "payment_method": "credit card",
            "monthly_charges": 1000000.00,
            "total_charges": 6000000.00,
            "feedback": "As an event planner, I've worked with many florists, but this one stands out from the crowd. For a recent high-profile corporate gala, they provided centerpieces that were nothing short of spectacular. The creativity and artistry in their designs elevated the entire event. They listened carefully to our theme and color scheme, then created arrangements that perfectly captured the essence of the evening. The flowers were fresh, vibrant, and lasted throughout the night, even under bright lights and air conditioning. What impressed me most was their flexibility - when we needed last-minute changes due to a shift in table arrangements, they accommodated without hesitation. Their team was professional, punctual, and a joy to work with. They arrived well before the event to set up and stayed until everything was perfect. The value for money was excellent, considering the high-quality results and the level of service provided. I've already booked them for several upcoming events, including a charity fundraiser and a product launch. It's rare to find a vendor that consistently exceeds expectations, but this florist does just that.",
            "topic": "general feedback"
        },
        "Sample 2": {
            "customer_id": "CUST002",
            "tenure": 24,
            "contract": "month-to-month",
            "payment_method": "electronic check",
            "monthly_charges": 750000.00,
            "total_charges": 1800000.00,
            "feedback": "I'm extremely disappointed with the quality of flowers in my recent order from this florist. When I first received the arrangement, I was initially impressed by its appearance. However, within just 24 hours, the flowers had already begun to wilt and lose their vibrancy. This rapid deterioration was shocking, especially considering the premium price I paid for what was supposed to be a high-quality product.\n\nMoreover, the arrangement I received didn't match the picture displayed on the website at all. The website showed a lush, full bouquet with a variety of colorful blooms, but what I got was a sparse arrangement with fewer flowers and less variety than advertised. This discrepancy between the advertised product and what was actually delivered feels misleading and has significantly diminished my trust in this florist.\n\nI've been a loyal customer for two years now, consistently ordering flowers for various occasions, but this experience has made me seriously reconsider my choice of florist. The combination of poor flower quality and misrepresentation of the product has left me feeling frustrated and undervalued as a customer. I expected much better from a service I've used and trusted for so long.",
            "topic": "product quality"
        },
        "Sample 3": {
            "customer_id": "CUST003",
            "tenure": 6,
            "contract": "two year",
            "payment_method": "bank transfer",
            "monthly_charges": 400000.00,
            "total_charges": 2400000.00,
            "feedback": "Avoid this florist at all costs! I ordered a birthday bouquet for my wife, and the experience was a disaster from start to finish. The website was glitchy and outdated, making the ordering process frustrating and time-consuming. Despite selecting a specific delivery date and time frame, they delivered a day early when no one was home, leaving the flowers outside in the heat without any notification. By the time we got to them, they were wilted and sad-looking, with petals already falling off. The arrangement bore little resemblance to the photo on their website - it was sparse, with fewer flowers than promised, and the colors didn't match at all. Some flowers were already dropping petals, clearly not fresh, and there was an unpleasant odor suggesting that some were starting to rot. The vase was a cheap plastic one instead of the glass vase advertised, and there was no card attached despite my having written a message during the order process. When I complained, their response was slow and unsympathetic. They offered to resend flowers, but at that point, the birthday had passed, and the moment was ruined. Their prices are high, but the quality and service don't match at all. This florist clearly doesn't value their customers or take pride in their work. The entire experience was stressful and disappointing, turning what should have been a lovely surprise into a source of frustration. A complete waste of money and a huge disappointment - I would give zero stars if I could.",
            "topic": "general feedback"
        },
        "Sample 4": {
            "customer_id": "CUST004",
            "tenure": 36,
            "contract": "two year",
            "payment_method": "mailed check",
            "monthly_charges": 600000.00,
            "total_charges": 2160000.00,
            "feedback": "I absolutely adore the wide variety of bouquets offered by this florist. Their commitment to providing diverse and creative floral arrangements has kept me a loyal customer for three years now. The seasonal collections, in particular, never fail to impress me with their innovative designs and carefully curated flower selections that perfectly capture the essence of each season.\n\nWhat truly sets this florist apart is the consistent quality of their flowers. Every bouquet I've received has been remarkably fresh, with blooms that are vibrant and long-lasting. It's a joy to watch the flowers slowly open over the course of a week or more, filling my home with beauty and fragrance. This longevity is a testament to the florist's dedication to sourcing the finest quality blooms and their expertise in proper flower care and handling.\n\nMoreover, I appreciate how the florist stays current with floral trends while also offering timeless classics. Whether I'm looking for a modern, avant-garde arrangement or a traditional bouquet, I always find something that suits my taste. The attention to detail in each arrangement, from the selection of complementary flowers to the artful composition, demonstrates a high level of skill and artistic vision. This consistent excellence in both variety and quality is why I remain a satisfied customer and why I wholeheartedly recommend this florist to friends and family.",
            "topic": "bouquet preferences"
        },
        "Sample 5": {
            "customer_id": "CUST005",
            "tenure": 18,
            "contract": "two year",
            "payment_method": "credit card",
            "monthly_charges": 550000.00,
            "total_charges": 990000.00,
            "feedback": "I cannot speak highly enough of the exceptional customer service provided by this florist. Recently, I found myself in a challenging situation where I needed to make last-minute changes to my flower order for a very important event. Feeling anxious about the tight timeline, I reached out to their customer service team, not really expecting much given the short notice.\n\nTo my absolute amazement, the team not only responded promptly but also displayed an incredible level of understanding and willingness to help. They listened attentively to my concerns and the changes I needed, showing genuine empathy for my situation. What impressed me most was their proactive approach - they didn't just passively take my requests, but actively suggested solutions and alternatives that I hadn't even considered, which ended up being perfect for my needs.\n\nThe level of dedication shown by the team went far beyond my expectations. They worked tirelessly to accommodate my changes, coordinating with their florists and delivery team to ensure everything would be perfect. Throughout the process, they kept me updated, alleviating my stress and giving me confidence that everything would work out. When the flowers arrived, right on time and exactly as discussed, I was overjoyed. The quality of the arrangement was superb, and it was clear that extra care had been taken to fulfill my modified request. This experience has solidified my loyalty to this florist. Their commitment to customer satisfaction, even in challenging circumstances, is truly commendable and sets them apart in the industry.",
            "topic": "customer service"
        }
    }

    # Radio buttons for sample selection (horizontal)
    selected_sample = st.radio("Select a sample", list(samples.keys()), index=0, horizontal=True)

    # Get the selected sample data
    sample_data = samples[selected_sample]

    with st.form(key="data"):
        col1, col2 = st.columns(2)
        with col1:
            customer_id = st.text_input("Customer ID", value=sample_data["customer_id"])
            tenure = st.number_input("Tenure (in months)", value=sample_data["tenure"])
        with col2:
            contract = st.selectbox(
                "contract", ['one year', 'month-to-month', 'two year'], index=['one year', 'month-to-month', 'two year'].index(sample_data["contract"])
            )
            payment_method = st.selectbox(
                "payment_method", ['credit card', 'electronic check', 'bank transfer', 'mailed check'], index=['credit card', 'electronic check', 'bank transfer', 'mailed check'].index(sample_data["payment_method"])
            )
        monthly_charges = st.number_input("Monthly charges", value=sample_data["monthly_charges"], step=1.00)
        total_charges = st.number_input("Total charges", value=sample_data["total_charges"], step=1.00)
        feedback = st.text_area(
            "Feedback", 
            value=sample_data["feedback"],
            height=300
        )
        topic = st.selectbox(
            "topic", ['bouquet preferences', 'delivery issues', 'general feedback', 'price complaints', 'delivery quality', 'product quality', 'customer service', 'price appreciation'], index=['bouquet preferences', 'delivery issues', 'general feedback', 'price complaints', 'delivery quality', 'product quality', 'customer service', 'price appreciation'].index(sample_data["topic"])
        )
        # Submit button
        submit = st.form_submit_button("🔘 Predict")

    if submit:
        data = {
            "customer_id": customer_id,
            "tenure": tenure,
            "contract": contract,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "feedback": feedback,
            "topic": topic
        }

        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Sentiment Analysis
        if feedback.strip() != "":
            status_text.text("Performing sentiment analysis...")
            for i in range(50):
                time.sleep(0.001)
                progress_bar.progress(i + 1)
            
            st.write("### 📊 Sentiment Analysis")
            st.markdown('---')
            labels, confidences = predict_sentiment([feedback])
            label = labels[0]
            confidence = confidences[0]
            st.write("**Feedback:**")
            st.write(feedback)
            if label == "Negative":
                st.error(f"Predicted Sentiment: **{label}**")
            else:
                st.success(f"Predicted Sentiment: **{label}**")
            st.info(f"Confidence: {confidence * 100:.2f}%")
            data['sentiment'] = label
        else:
            st.warning("Feedback is empty, skipping sentiment analysis.")
            data['sentiment'] = None

        # Convert data to DataFrame for churn prediction
        data = pd.DataFrame([data])

        # Load customer churn model
        status_text.text("Loading churn prediction model...")
        for i in range(50, 75):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        with open('model.pkl', 'rb') as file_1:
            classification = pickle.load(file_1)

        status_text.text("Predicting churn...")
        for i in range(75, 100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        churn = classification.predict(data)
        churn_bool = churn.tolist() 

        # Clear the progress bar and status text
        progress_bar.empty()
        status_text.empty()

        # Display churn prediction result
        st.write("### 🕵️♂️ Prediction Results: ")
        st.markdown('---')
        st.write("")  # Add a blank line for spacing
        
        if all (not element for element in churn_bool):
            st.success("🙆 Customer is Not Gonna Churn")
            st.image('sss.png')
            st.balloons()
            st.write('### 💭 Feedback To Marketing Team: ')
            st.warning("""
            The customer is not going to churn as long as we continue delivering consistent value, addressing their needs effectively, and providing exceptional service that keeps them satisfied and engaged. By maintaining open communication, offering personalized experiences, and responding quickly to any issues, we can ensure their loyalty and reduce the likelihood of churn.
            """)
        else:
            if any(element for element in churn_bool):
                st.error("🏃 **Customer is Gonna Churn!!**")
                st.image('8908ec58-057d-4bfe-9ce5-74322486859a.png')
                st.write('')  # Add a blank line for spacing
                st.write('### 💭 Feedback To Marketing Team: ')
                st.error("""
                        - **The customer** is likely to churn if we don't take immediate steps to improve our **service quality**. Consistently providing subpar service will push them to seek out competitors who can meet their expectations more reliably, resulting in a loss of **long-term loyalty and revenue**.

                        - Without implementing a more effective **retention strategy**, our customers are at high risk of churning. It's crucial that we personalize our **offerings, provide timely incentives, and enhance customer communication** to keep them **engaged and loyal** to our brand.

                        - The recent product changes have led to growing dissatisfaction among customers, increasing the likelihood of churn. To prevent this, we must swiftly **address their concerns, re-evaluate the changes, and ensure that future updates align with customer expectations** to regain their trust.

                        - If we fail to address customer concerns promptly, we risk driving them away. Providing timely and empathetic responses is essential to **resolving issues, maintaining trust, and ensuring that customers feel valued**, which is critical to reducing churn.

                        - Without a personalized engagement approach, customers are more likely to churn, as they will **feel disconnected** from our brand. Tailoring our **communication and offers** based on **individual customer preference**s can significantly improve **satisfaction and foster long-term loyalty**.
                        """)
            else: st.write("�� **No churn prediction available**")

if __name__ == "__main__":
    run()