from keras import layers
from keras import models
from datetime import datetime
import numpy as np
import config


def distance(pre_fiber, exp_fiber):
    # get points(x,y) for the pred and exp fiber
    pre_fiber_re = pre_fiber.reshape(config.num_points, -1)[:, :2].flatten()
    exp_fiber_re = exp_fiber.reshape(config.num_points, -1)[:, :2].flatten()
    return np.sqrt(np.sum(np.square(pre_fiber_re - exp_fiber_re)))


def sem_model(x_train, y_train):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', use_bias=False,
                            input_shape=(config.img_size, config.img_size, 1)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(96, (3, 3), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(512, (3, 3), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(config.num_fiber * config.num_points * 3))
    model.summary()

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])

    num_epochs = config.num_epochs
    samples = config.num_img
    num_objects = config.num_fiber
    flipped_train_y = np.array(y_train)
    flipped = np.zeros((samples, num_epochs))
    dists_epoch = np.zeros((samples, num_epochs))
    mses_epoch = np.zeros((samples, num_epochs))

    for epoch in range(num_epochs):
        print('Epoch', epoch)
        model.fit(x_train, flipped_train_y, epochs=1, batch_size=1, validation_split=0.2, verbose=2)
        pred_y = model.predict(x_train)

        for sample, (pred_fibers, exp_fibers) in enumerate(zip(pred_y, flipped_train_y)):
            pred_fibers = pred_fibers.reshape(num_objects, -1)
            exp_fibers = exp_fibers.reshape(num_objects, -1)

            dists = np.zeros((num_objects, num_objects))
            mses = np.zeros((num_objects, num_objects))
            for i, exp_fiber in enumerate(exp_fibers):
                for j, pred_fiber in enumerate(pred_fibers):
                    dists[i, j] = distance(exp_fiber, pred_fiber)
                    mses[i, j] = np.mean(np.square(exp_fiber - pred_fiber))

            new_order = np.zeros(num_objects, dtype=int)

            for i in range(num_objects):
                # Find pred and exp fibers with minimum distance and assign them to each other
                ind_exp_fiber, ind_pred_fiber = np.unravel_index(dists.argmin(), dists.shape)
                dists_epoch[sample, epoch] += dists[ind_exp_fiber, ind_pred_fiber]
                mses_epoch[sample, epoch] += mses[ind_exp_fiber, ind_pred_fiber]

                mses[ind_exp_fiber] = 10000000
                mses[:, ind_pred_fiber] = 10000000
                new_order[ind_pred_fiber] = ind_exp_fiber

            flipped_train_y[sample] = exp_fibers[new_order].flatten()

            flipped[sample, epoch] = 1. - np.mean(new_order == np.arange(num_objects, dtype=int))
            dists_epoch[sample, epoch] /= num_objects
            mses_epoch[sample, epoch] /= num_objects

        print('Flipped {} % of all elements'.format(np.mean(flipped[:, epoch]) * 100.))
        print('Mean dist: {}'.format(np.mean(dists_epoch[:, epoch])))
        print('Mean mse: {}'.format(np.mean(mses_epoch[:, epoch])))
        print()

    now = datetime.now()
    model.save('model' + str(now)[-6:] + '.h5')

    return model

