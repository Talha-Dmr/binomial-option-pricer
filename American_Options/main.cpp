#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "CLI11.hpp"  // CLI11 tek baþýna header'dýr

enum class OptionType { CALL, PUT };

struct Option {
    double S;
    double K;
    double T;
    double sigma;
    double r;
    int N;
    OptionType type;
};

struct BinomialParams {
    double u, d, p, dt;
};

BinomialParams compute_binomial_params(const Option& opt) {
    BinomialParams params;
    params.dt = opt.T / opt.N;
    params.u = std::exp(opt.sigma * std::sqrt(params.dt));
    params.d = 1.0 / params.u;
    params.p = (std::exp(opt.r * params.dt) - params.d) / (params.u - params.d);
    return params;
}

std::vector<double> terminal_payoff(const Option& opt, const BinomialParams& params) {
    std::vector<double> values(opt.N + 1);
    double ST = opt.S * std::pow(params.d, opt.N);
    for (int j = 0; j <= opt.N; ++j) {
        if (opt.type == OptionType::CALL)
            values[j] = std::max(0.0, ST - opt.K);
        else
            values[j] = std::max(0.0, opt.K - ST);
        ST *= params.u / params.d;
    }
    return values;
}

double binomial_tree_american(const Option& opt) {
    BinomialParams params = compute_binomial_params(opt);
    std::vector<double> values = terminal_payoff(opt, params);
    for (int step = opt.N - 1; step >= 0; --step) {
        double S = opt.S * std::pow(params.d, step);
        for (int i = 0; i <= step; ++i) {
            double continuation = std::exp(-opt.r * params.dt) *
                (params.p * values[i + 1] + (1.0 - params.p) * values[i]);
            double exercise = (opt.type == OptionType::CALL)
                ? std::max(0.0, S - opt.K)
                : std::max(0.0, opt.K - S);
            values[i] = std::max(continuation, exercise);
            S *= params.u / params.d;
        }
    }
    return values[0];
}

int main(int argc, char** argv) {
    CLI::App app{ "Binomial Tree American Option Pricer" };

    double S, K, T, sigma, r;
    int N;
    std::string type_str = "call";

    app.add_option("-S,--spot", S, "Spot price")->required();
    app.add_option("-K,--strike", K, "Strike price")->required();
    app.add_option("-T,--maturity", T, "Time to maturity (years)")->required();
    app.add_option("--sigma", sigma, "Volatility (e.g. 0.2)")->required();
    app.add_option("-r,--rate", r, "Risk-free rate (e.g. 0.05)")->required();
    app.add_option("-N,--steps", N, "Number of binomial tree steps")->required();
    app.add_option("--type", type_str, "Option type: call or put (default: call)");

    CLI11_PARSE(app, argc, argv);

    Option opt;
    opt.S = S;
    opt.K = K;
    opt.T = T;
    opt.sigma = sigma;
    opt.r = r;
    opt.N = N;
    // type_str'i küçük harfe çevir, güvenli hale getir:
    std::transform(type_str.begin(), type_str.end(), type_str.begin(), ::tolower);
    if (type_str == "call")
        opt.type = OptionType::CALL;
    else if (type_str == "put")
        opt.type = OptionType::PUT;
    else {
        std::cerr << "Option type must be 'call' or 'put'!" << std::endl;
        return 1;
    }

    double price = binomial_tree_american(opt);

    std::cout << "\n==> American Option Price: " << price << std::endl;

    return 0;
}
