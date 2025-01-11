# SerVLM  == 🧀VLM


### Install uv
If `uv` is not installed, get it on mac / linux by running:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

for windows see https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2



### Install Dependencies

```
uv sync
```

## Start Server

```
uv run -- fastapi dev servlm
```

It should start running on http://127.0.0.1:8000/

The docs are available at http://127.0.0.1:8000/docs




## Run notebook with demo of client

A demo of the client is available in the `./playground.ipynb` notebook

to run jupyter lab use:
```
uv run -- jupyter lab
```