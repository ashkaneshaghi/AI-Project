from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def model_evaluation(model, x_train, y_train, x_test, y_test, scale=False, poly_deg=False):
    if poly_deg:
        poly = PolynomialFeatures(degree=poly_deg, include_bias=False)
        poly_x_train = poly.fit_transform(x_train)
        poly_x_test = poly.transform(x_test)
        if scale:
            scaler = StandardScaler()
            scale_x_train = scaler.fit_transform(poly_x_train)
            scale_x_test = scaler.transform(poly_x_test)
        else:
            scale_x_train = poly_x_train
            scale_x_test = poly_x_test
    else:
        if scale:
            scaler = StandardScaler()
            scale_x_train = scaler.fit_transform(x_train)
            scale_x_test = scaler.transform(x_test)
        else:
            scale_x_train = x_train
            scale_x_test = x_test

    # Training
    model = model.fit(scale_x_train, y_train)
    # Predicting
    y_pred = model.predict(scale_x_test)

    training_score = model.score(scale_x_train, y_train)
    testing_score = model.score(scale_x_test, y_test)

    return model, y_pred, training_score, testing_score
