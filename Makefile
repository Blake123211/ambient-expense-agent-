.PHONY: playground generate-traces grade

playground:
	adk web .

generate-traces:
	python tests/eval/generate_traces.py

grade:
	agents-cli grade --config tests/eval/eval_config.yaml
