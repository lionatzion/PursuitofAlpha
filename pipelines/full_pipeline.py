import subprocess

def main():
    subprocess.run(["python","pipelines/train_pipeline.py"])
    subprocess.run(["python","pipelines/backtest_pipeline.py"])

if __name__=="__main__":
    main()
