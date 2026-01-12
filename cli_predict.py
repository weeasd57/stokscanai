import runpy


if __name__ == "__main__":
    # Run the CLI entrypoint located in api/
    runpy.run_path("api/cli_predict.py", run_name="__main__")
