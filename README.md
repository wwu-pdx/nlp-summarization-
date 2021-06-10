# nlp-summarization
* Even the least optimized bart outpeform GTP2 and Bert
* GTP2 does better than Bert
* Need to investigate the cuda issue and how to move str across GPU and CPU
* Strings could take much longer to process than images, because images are essenitally represented in numbers

|    CNN/DM          | R1                 |          R2       |      RL             |
|    ----------      |   ------------     |   --------------    |   ---------------    |
| cnn bart_pipe_tune | 0.3741738842701064  | 0.1694944817545712  | 0.3741738842701064  |
| cnn bart_token     | 0.3546711358693531  | 0.15793400970373578 | 0.3546711358693531  |
| cnn bart           | 0.34696367113830123 | 0.1377183238047612  | 0.34696367113830123 |
| cnn gpt2           | 0.24448375202981537 | 0.07376025731186839 | 0.24448375202981537 |
| cnn bert           | 0.27108347899912943 | 0.08703011151816702 | 0.27108347899912943 |


