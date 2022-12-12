### For custom dataset
- Put the image dataset inside `Keras-GAN/wgan/dataset/` folder
- Update the `self.img_rows`, `self.img_cols`, `self.channels` value.
- Update the following lines inside `build_generator()` function.
```python
model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
model.add(Reshape((7, 7, 128)))
```
- Change the `(X_train, _), (_, _) = mnist.load_data()` with `X_train = load_image()`