from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.express as pex
import plotly.io as pio
import matplotlib.pyplot as plat
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    x = np.random.normal(10, 1, 1000)
    univariateGaussian = UnivariateGaussian()
    univariateGaussian.fit(x)
    print(str(univariateGaussian.mu_)+", "+str(univariateGaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    loss_arr = []
    univariateGaussian = UnivariateGaussian()
    for i in range(10, 1010, 10):
        univariateGaussian.fit(x[:i])
        i_loss = np.abs(univariateGaussian.mu_ - 10)
        loss_arr.append(i_loss)

    graph = pex.line(x=range(10, 1010, 10),
                     y=np.asarray(loss_arr),
                     labels=dict(x="Sample size", y="Loss of the expectation"),
                     title="Absolute distance between the estimated- and true value of the expectation")
    graph.show()
    # graph.write_html('first_figure.html', auto_open=True)

    # Question 3 - Plotting Empirical PDF of fitted model
    plat.scatter(x, univariateGaussian.pdf(x))
    plat.ylabel("PDF")
    plat.xlabel("SAMPLES")
    plat.title("Empirical PDF of samples")
    plat.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    MU = np.array([0, 0, 4, 0])
    COV = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])

    X = np.random.multivariate_normal(MU, COV, 1000)
    multivariateGaussian = MultivariateGaussian()
    multivariateGaussian.fit(X)

    print(multivariateGaussian.mu_)
    print(multivariateGaussian.cov_)


    # Question 5 - Likelihood evaluation
    heatmap = []
    args_range = np.linspace(-10, 10, 200)
    for f1 in args_range:
        arr = []
        for f3 in args_range:
            log_likelihood_i = multivariateGaussian.log_likelihood(np.asarray([f1, 0, f3, 0]).T, COV, X)
            arr.append(log_likelihood_i)
        heatmap.append(arr)

    graph = pex.imshow(heatmap, x=args_range, y=args_range, labels=dict(x="f1", y="f3", color="log-likelihood"),
                       title="Log Likelihood")
    graph.show()


    # Question 6 - Maximum likelihood
    max_likelihood_coords = np.where(heatmap == np.max(heatmap))
    f1 = args_range[max_likelihood_coords[0][0]]
    f3 = args_range[max_likelihood_coords[1][0]]
    print(f"{'{:.3f}'.format(f1)} {'{:.3f}'.format(f3)}")



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
