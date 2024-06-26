# Template ML Proj
This project aims at helping building your own Machine Learning code with some basic coding frame, such as different functional files and basic logical training process.
## File Structure
```
├── README.md               // Description
├── __init__.py             // Python init file
├── datawork.py             // You can put your `data processing` func here
├── model.py                // You can put your `model` here
├── training.py             // You can put your `training` func here
├── main.py                 // The main func to start your whole proj
├── config.yaml             // Define your hyperparameters
├── data                    // Put your data here
│    ├── covid_train.csv    // Train data
│    └── covid_test.csv     // Test data
├── logs                    // Record your model training (Modify in config.yaml if no need logging)
│    └── test_0             // 'test' training record
│         ├── checkpoint_best_epoch.pkl     // The saved model in best epoch
│         ├── checkpoint_{epoch}_epoch.pkl  // The saved model in `epoch` epoch
│         └── checkpoint_last_epoch.pkl     // The saved model in last epoch
└── utils                   // Put your utils funcs here
     └── modelsave.py       // Model saving func

```
## Acknowledge
Special thanks to Prof. Lee Hong-Yi from Taiwan Univ for his extrordinary ML lessons. The demo training data is from his [ML 2023 lesson](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php).
