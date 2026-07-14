text = "I didn't like the movie, it was disappointing."
#Perform sentiment analysis usig a pretrained model
classifier = pipelne('sentiment-analysis')

#Extract predicted label and score
predicted_label = result[0]['label']
confidence_score = result[0]['score']

#check if the predicted sentiment is negative
if predicted_label == 'NEGATIVE':
    print(f"The sentiment analysis indicates a negative sentiment for the textt:")
    print(f"Confidence score:{confidence_score:.4f}")
else:
    print(f"The sentiment analysis indicates a positive sentiment for the text:'{text}'")
    print(f"Confidence score:{confidence_score:.4f}")