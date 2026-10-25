PY ?= python

test:
	$(PY) -m pytest -q

data:
	$(PY) scripts/prepare_data.py

index:
	$(PY) scripts/build_index.py

eval:
	$(PY) scripts/run_eval.py

app:
	streamlit run app/streamlit_app.py
