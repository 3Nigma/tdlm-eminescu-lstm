# Introduction

This repo contains a LSTM tensorflow model along with its **deeplearn.js** usage
example to load and display in a browser a chunk of 512 characters every 10 secs.

The training data was taken from a compilation of poems written by Mihai Eminescu, a
romanian poet. You can find the original data [here](http://www.gutenberg.org/ebooks/35323).

Python sources can be found in `src/python` while the web code can be inspected
in `src/web`.

It was presented at the Timisoara Deep Learning Meetup on LSTMs on 1 Februrary 2018.
Slides can be accessed [here](https://docs.google.com/presentation/d/1NZ52WiS6d5MqC9zPg3D1EXd68L2HCxHTelDPRYEM42c).

# Prerequisites
A Linux distro is required having the following tools installed:
* [Python](https://www.python.org/downloads/)
* [Tensorflow](https://www.tensorflow.org/install/) library (GPU vaiant preferably)
* [NodeJS](https://nodejs.org/en/download/)
* [Yarn](https://yarnpkg.com/lang/en/docs/install/) package manager 

# Usage
First you have to install the dependencies. You do this with:
```bash
yarn prep
```
Then you need to train the network. You can tweak the hyperparameters from
`src/python/params.py`. To start the training, you type:
```bash
yarn train
```
Next you need to export the network's learned parameters along with compiling
and deploying the web typescript sources (`src/web/main.ts`). You do this by
typing:
```bash
yarn deploy
```

Having this done, you can see the deployed files in `build` folder.

To start serving the page and have it viewable from the browser, just type:
```bash
yarn start-server
```

wait 10 seconds and you shall see the first batch of text generated.

You can also do a `yarn clean` which deletes the `build` folder along with all
the `node_modules` leaving you with a clean source base. Keep in mind that if you
do this, you will have to redo all the previous steps to access the page again.

# License
You are free to use the info/code you see here however you want with no restrictions.

## Happy learning! :beer:
