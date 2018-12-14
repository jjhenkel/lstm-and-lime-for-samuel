FROM gw000/keras:2.1.4-py3

COPY labels.py /app/
COPY predict_one.py /app/
COPY models/best-1.h5 /app/
COPY assets/tokenizer.pkl /app/

CMD [ "/app/predict_one.py" ]
ENTRYPOINT [ "python3" ]
