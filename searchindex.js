Search.setIndex({"docnames": ["index", "kernel/MMD", "kernel/RKHS", "kernel/base", "kernel/covariance_operators", "mathmodel/bregman_divergence", "mathmodel/conjugate_dist", "mathmodel/entropy", "mathmodel/statistic_model", "matrix/base", "matrix/decomposition", "matrix/matrixdiff", "matrix/matrixoper", "matrix/special_matrix", "matrix/subspace", "matrix/vectorspace", "ml/EM", "ml/biClustering", "ml/dimension_reduce", "ml/neuro_representation", "optimization/convex_neq_solve", "optimization/convex_prob", "optimization/convex_solve"], "filenames": ["index.rst", "kernel/MMD.md", "kernel/RKHS.md", "kernel/base.md", "kernel/covariance_operators.md", "mathmodel/bregman_divergence.md", "mathmodel/conjugate_dist.md", "mathmodel/entropy.md", "mathmodel/statistic_model.md", "matrix/base.md", "matrix/decomposition.md", "matrix/matrixdiff.md", "matrix/matrixoper.md", "matrix/special_matrix.md", "matrix/subspace.md", "matrix/vectorspace.md", "ml/EM.md", "ml/biClustering.md", "ml/dimension_reduce.md", "ml/neuro_representation.md", "optimization/convex_neq_solve.md", "optimization/convex_prob.md", "optimization/convex_solve.md"], "titles": ["\u673a\u5668\u5b66\u4e60\u57fa\u7840", "<span class=\"section-number\">3. </span>\u6700\u5927\u5747\u503c\u5dee", "<span class=\"section-number\">2. </span>\u518d\u751f\u6838\u5e0c\u5c14\u4f2f\u7279\u7a7a\u95f4", "<span class=\"section-number\">1. </span>\u6838\u51fd\u6570\u57fa\u7840", "<span class=\"section-number\">4. </span>\u534f\u65b9\u5dee\u7b97\u5b50", "<span class=\"section-number\">2. </span>Bregman divergence", "<span class=\"section-number\">4. </span>\u5171\u8f6d\u5206\u5e03", "<span class=\"section-number\">3. </span>\u4fe1\u606f\u71b5", "<span class=\"section-number\">1. </span>\u4e0d\u786e\u5b9a\u6a21\u578b", "<span class=\"section-number\">1. </span>\u77e9\u9635\u6027\u80fd\u6307\u6807", "<span class=\"section-number\">7. </span>\u77e9\u9635\u5206\u89e3", "<span class=\"section-number\">4. </span>\u77e9\u9635\u5fae\u5206", "<span class=\"section-number\">2. </span>\u77e9\u9635\u8fd0\u7b97", "<span class=\"section-number\">5. </span>\u7279\u6b8a\u77e9\u9635", "<span class=\"section-number\">6. </span>\u5b50\u7a7a\u95f4\u5206\u6790", "<span class=\"section-number\">3. </span>\u5411\u91cf\u7a7a\u95f4", "<span class=\"section-number\">3. </span>EM\u7b97\u6cd5\u6982\u8ff0", "<span class=\"section-number\">1. </span>BiClustering", "<span class=\"section-number\">1. </span>\u6570\u636e\u964d\u7ef4", "<span class=\"section-number\">2. </span>\u8868\u793a\u5b66\u4e60", "<span class=\"section-number\">3. </span>\u4f18\u5316\u95ee\u9898\u6c42\u89e3(2)", "<span class=\"section-number\">1. </span>\u51f8\u4f18\u5316\u95ee\u9898", "<span class=\"section-number\">2. </span>\u4f18\u5316\u95ee\u9898\u6c42\u89e3(1)"], "terms": {"lle": 0, "autoencod": 0, "em": 0, "biclust": 0, "biclustr": 0, "via": 0, "singlular": 0, "valu": [0, 5], "decomposit": [0, 18], "hilbert": [0, 1, 3, 15], "mean": [0, 5, 18], "embed": [0, 18], "maximum": 0, "discrep": 0, "schmidt": 0, "newton": [0, 20], "lagrangian": [0, 18, 22], "alm": 0, "admm": 0, "hadamard": [0, 9, 11], "kroneck": [0, 11], "jacobian": 0, "hessian": [0, 20, 22], "hermitian": [0, 14], "qr": [0, 14], "bregman": 0, "diverg": 0, "squar": 0, "euclidean": 0, "distanc": 0, "refer": 0, "python": 0, "gamma": [0, 3, 4], "beta": [0, 2, 4, 11, 22], "mathcal": [1, 2, 3, 4, 8, 15, 16, 18, 19, 21, 22], "borel": 1, "mu_p": 1, "mathbb": [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22], "_p": [1, 12], "varphi_i": 1, "sim": [1, 4, 6, 8, 19], "langl": [1, 2, 3, 4, 5, 9, 10, 15, 18, 20, 21], "mu_q": 1, "rangle_": [1, 2, 3, 4], "varphi": [1, 4], "cdot": [1, 2, 3, 4, 9, 10, 11, 12, 15, 18, 20, 21, 22], "begin": [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 16, 18, 20, 21, 22], "bmatrix": [1, 4, 10, 12, 14, 18, 20], "end": [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 16, 18, 20, 21, 22], "quad": [1, 2, 3, 4, 5, 6, 8, 9, 10, 14, 18, 19, 20, 21, 22], "top": [1, 3, 4, 9, 10, 11, 12, 13, 14, 15, 18, 20, 22], "ax": [1, 9, 10, 11, 12, 13, 14, 18, 20, 21, 22], "bx": [1, 20, 22], "left": [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21, 22], "right": [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21, 22], "triangleq": [1, 3, 4, 9, 11, 16, 18, 19, 22], "varphi_1": 1, "varphi_2": 1, "in": [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22], "sqrt": [1, 2, 3, 4, 8, 9, 10, 15, 18], "infti": [1, 2, 3, 4, 6, 7, 9, 15, 18, 19, 20, 21], "exist": [1, 15], "t_pf": 1, "foral": [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 18, 21], "split": [1, 2, 3, 4, 6, 7, 9, 10, 11, 14, 16, 18, 20, 21, 22], "le": [1, 2, 3, 4, 6, 8, 9, 10, 13, 15, 18, 20, 21, 22], "vert": [1, 2, 3, 4, 5, 10, 15, 18, 19, 20], "vert_": [1, 2, 4, 15], "riesz": [1, 2, 4], "lambda_": [1, 9, 18, 20], "t_p": 1, "g_a": [1, 2], "af": [1, 2], "lambda_a": 1, "textrm": [1, 2, 3, 4, 5, 6, 7, 9, 18, 20], "underbrac": [1, 2, 3, 4, 5, 6, 18], "_q": [1, 12], "within": 1, "distrib": 1, "similar": [1, 5, 17], "cross": 1, "rangl": [1, 2, 3, 4, 5, 9, 10, 15, 18, 20, 21], "d_": [1, 7, 11, 18, 19], "sup_": [1, 2], "name": 1, "formula": 1, "condit": [1, 3, 4, 5, 7], "dudley": 1, "bl": 1, "vert_l": 1, "sup": [1, 2, 4, 20, 21], "rho": [1, 20], "neq": [1, 7, 8, 9, 10, 11, 12, 18, 22], "wasserstein": 1, "inf_": 1, "mu": [1, 4, 15, 16, 20, 21], "int": [1, 2, 18, 22], "and": [1, 3, 5, 17, 20], "is": [1, 2, 3, 5, 17, 18], "separ": [1, 4], "rkhs": 1, "frac": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 16, 18, 19, 20, 21, 22], "box": [1, 2, 4, 5, 18], "approx": [1, 18, 20, 22], "frac1m": 1, "sum_": [1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 16, 18, 19, 20, 21, 22], "x_i": [1, 2, 4, 5, 6, 9, 11, 18], "frac1n": [1, 4, 8, 16, 18], "x_j": [1, 2, 4, 6, 9, 11, 14], "x_": [1, 6, 11, 13, 18, 20, 22], "mn": [1, 9, 11, 12, 18], "function": [2, 5, 16, 18], "rightarrow": [2, 3, 4, 6, 11, 12, 14, 15, 18, 19, 20, 21, 22], "subseteq": [2, 15, 18], "int_": [2, 15], "x_0": 2, "x_1": [2, 3, 4, 6, 9, 11, 12, 18, 22], "dx": [2, 15], "ds": 2, "dy": [2, 11, 18], "int_0": [2, 6], "mgi": 2, "mv": 2, "2gi": 2, "dt": [2, 6], "hat": [2, 3, 4, 12, 16, 18, 19, 20], "calculus": 2, "of": [2, 3, 5, 7, 16, 17, 18], "variat": 2, "epsilon": [2, 15, 20, 21, 22], "eta": [2, 20], "delta": [2, 4, 6, 18, 20, 22], "partial": [2, 9, 11, 18, 21, 22], "boundari": 2, "inf": [2, 20], "qquad": [2, 4, 18, 21], "euler": 2, "lagrang": [2, 20], "int_a": 2, "f_i": [2, 18, 20, 21], "_y": [2, 4], "isbn": 2, "9787040427806": 2, "kx": 2, "frac12": [2, 6, 9, 11, 18, 20, 21, 22], "2r": 2, "sin": [2, 18], "theta": [2, 16, 19, 21, 22], "cos": [2, 18, 22], "array": [2, 7, 9, 10, 11, 12, 16, 18, 20, 21, 22], "homogen": 2, "alpha": [2, 3, 4, 6, 11, 22], "addit": [2, 3], "ag": 2, "subset": [2, 17], "banach": [2, 15], "l_p": [2, 3, 9], "a_g": 2, "norm": [2, 4, 9, 15, 18, 22], "linear": [2, 4], "space": [2, 3, 14, 15], "over": [2, 5], "itself": 2, "f_0": [2, 20, 21], "impli": 2, "af_0": 2, "f_1": [2, 3, 4, 20, 21], "f_2": [2, 3, 4, 21], "oper": 2, "bound": 2, "lambda": [2, 4, 6, 9, 10, 18, 20, 21, 22], "one": [2, 5, 11, 18], "point": [2, 5, 18, 20], "algebra": 2, "dual": [2, 20], "topolog": [2, 18], "represent": 2, "dirac": 2, "evalu": [2, 5], "let": 2, "be": [2, 3, 5, 18], "defin": [2, 3, 5], "on": [2, 3, 5], "non": [2, 5], "empti": 2, "set": 2, "for": [2, 3, 5, 9, 18], "fix": 2, "map": [2, 3, 18], "delta_x": 2, "call": 2, "the": [2, 3, 5, 16, 18], "at": [2, 5], "reproduc": 2, "kernel": [2, 3, 14, 18], "coroallari": 2, "lim_": [2, 6, 15], "f_n": 2, "l_2": [2, 9], "q_n": 2, "2n": [2, 9, 12, 18], "kappa": [2, 3], "time": [2, 3, 9, 10, 11, 12, 13, 14, 15, 18, 19], "properti": 2, "cauchi": [2, 3, 9, 15], "schwarz": [2, 3], "inequ": 2, "f_": [2, 5, 11, 18, 22], "delta_": [2, 11, 19], "anti": 2, "isometri": 2, "ge": [2, 3, 5, 7, 9, 10, 15, 16, 21], "a_i": [2, 3], "a_ia_jh": 2, "a_j": 2, "stricklli": 2, "posit": 2, "definit": 2, "lemma": [2, 3], "phi": [2, 3, 4, 18, 19, 20], "corollari": 2, "moor": 2, "aronszajn": 2, "xor": 3, "pmb": [3, 5, 8, 9, 10, 12, 14, 15, 16, 18, 19, 20, 21], "x_2": [3, 4, 9, 12, 18, 22], "x_1x_2": 3, "z_1": 3, "z_1z_2": 3, "z_2": 3, "2z_1": 3, "2z_2": 3, "2x_1x_2z_1z_2": 3, "x_1z_1": 3, "x_2z_2": 3, "trick": 3, "inner": [3, 15], "product": [3, 4, 15], "alpha_1": [3, 22], "alpha_2": 3, "if": [3, 5, 7, 16, 18, 22], "onli": [3, 5], "hilber": [3, 15], "which": [3, 5, 17], "an": 3, "along": 3, "with": [3, 17, 18], "technic": 3, "sum": [3, 7, 16, 18], "are": [3, 5], "k_1": 3, "k_2": 3, "between": [3, 5], "tild": [3, 4, 8, 14, 18], "_1": [3, 4, 8, 9, 10, 11, 12, 14, 18, 22], "_2": [3, 4, 9, 10, 18, 22], "polynomi": 3, "ell_p": 3, "ell_2": 3, "phi_i": 3, "th": 3, "taylor": [3, 22], "seri": 3, "expans": 3, "may": 3, "use": [3, 5, 18, 22], "to": [3, 5, 17, 18], "that": [3, 5], "have": [3, 5], "infinityli": 3, "mani": 3, "featur": [3, 18], "a_n": 3, "all": [3, 5], "exampl": 3, "exponenti": 3, "exp": [3, 11, 12, 18], "gaussian": 3, "normal": [3, 18], "2xx": 3, "la_i": 3, "_i": [3, 8, 9, 10, 11, 14, 16, 18, 19, 22], "a_ia_j": 3, "_j": [3, 4, 9, 10, 14, 18, 22], "phi_": 3, "frac1l": 3, "frac2l": 3, "k11": [3, 18], "11": [3, 4, 9, 11, 12, 15, 18, 22], "p_": [3, 4, 18], "bot": [3, 14], "ww": 3, "e_i": 4, "f_j": [4, 18], "compact": 4, "hs": 4, "lf_j": 4, "schmidti": 4, "hibert": 4, "mf_j": 4, "parsev": 4, "ident": 4, "alpha_i": [4, 22], "beta_": 4, "e_": [4, 18, 19], "tensor": 4, "ab": [4, 9, 11, 12, 13, 15], "lb": 4, "sum_j": [4, 9, 14, 15, 18], "kb": 4, "psi": 4, "xy": [4, 18], "xyg": 4, "c_": [4, 11, 18], "mu_x": 4, "mu_i": 4, "form": 4, "t_": [4, 8, 18], "mapsto": 4, "weaker": 4, "jensen": 4, "cov": 4, "_x": 4, "constrain": 4, "limits_": [4, 16, 18, 19, 20, 21, 22], "varphi_j": 4, "g_j": 4, "phi_j": 4, "vdot": [4, 9, 10, 11, 12, 18, 20], "g_1": 4, "g_2": 4, "y_i": [4, 5, 6], "emp": 4, "empir": 4, "gamma_": 4, "max": [4, 9, 16, 20, 21], "ij": [4, 9, 11, 12, 13, 15, 18], "y_j": [4, 16], "proof": 4, "sum_i": [4, 7, 8, 9, 15, 18, 21], "i_n": 4, "xhi": 4, "x_n": [4, 6, 9, 11, 15, 18], "y_1": [4, 6], "y_2": 4, "y_n": 4, "xh": 4, "yh": 4, "hxx": 4, "2_": 4, "surpris": 5, "result": 5, "due": 5, "banerje": 5, "gou": 5, "wang": 5, "2005": 5, "follow": 5, "you": 5, "some": 5, "abstract": 5, "way": 5, "measur": 5, "ani": 5, "two": [5, 17], "choic": 5, "distribut": 5, "minimis": 5, "averag": 5, "other": [5, 11], "then": 5, "your": 5, "must": 5, "member": 5, "famili": 5, "sed": 5, "can": 5, "repres": 5, "as": [5, 7, 16, 18, 22], "obvious": 5, "deriv": 5, "now": 5, "we": 5, "take": 5, "look": 5, "term": 5, "tangent": 5, "line": 5, "this": 5, "whole": 5, "express": 5, "just": 5, "differ": 5, "so": 5, "far": 5, "constraint": 5, "place": 5, "it": 5, "negat": [5, 22], "possibl": 5, "equival": 5, "alway": 5, "sit": 5, "abov": 5, "its": 5, "nabla": [5, 18, 20, 21, 22], "being": 5, "convex": 5, "d_f": 5, "has": [5, 18], "stuctur": 5, "by": 5, "simpli": 5, "choos": 5, "like": 5, "precis": 5, "guarante": 5, "they": 5, "kullback": 5, "leibler": 5, "kl": [5, 7, 11, 18], "np_i": 5, "log": [5, 7, 11, 16, 18, 20], "p_i": 5, "meet": 5, "tag": [6, 21, 22], "pi": [6, 16, 22], "chi": [6, 8], "dirichlet": 6, "mathrm": [6, 9, 11, 12, 13, 14, 20, 21], "ga": 6, "shape": [6, 7, 18], "rate": 6, "poisson": 6, "poissson": 6, "poi": 6, "10": [6, 15, 18, 22], "01": [6, 7, 22], "prod_": [6, 9, 16], "pmatrix": [6, 10, 18], "y_m": 6, "m_1": 6, "m_2": 6, "textbf": 6, "binomcount": 6, "shannon": 7, "1948": 7, "mathemat": 7, "theori": 7, "communic": 7, "cap": [7, 14], "frac1p": 7, "sum_x": 7, "cross_entropi": 7, "code": [7, 9, 16], "utf": [7, 16], "import": [7, 16, 18, 22], "numpi": [7, 16, 18, 22], "np": [7, 16, 18, 22], "from": [7, 18], "matlib": 7, "scipi": 7, "stat": 7, "entropi": 7, "def": [7, 16, 18, 22], "mutual_info": 7, "pxi": 7, "ndim": 7, "rais": 7, "except": 7, "hx": 7, "axi": [7, 18], "hx_i": 7, "cdh": 7, "return": [7, 16, 18, 22], "px": [7, 14, 18], "none": 7, "pk": 7, "check": 7, "prob": [7, 16], "py": [7, 18], "py_rep": 7, "repmat": 7, "px_i": 7, "jointh": 7, "len": [7, 16, 18], "__name__": [7, 16, 18, 22], "__main__": [7, 16, 18, 22], "65": 7, "25": 7, "07": 7, "03": 7, "05": 7, "ent": 7, "ent2": 7, "print": [7, 16, 18, 22], "84": 7, "02": 7, "06": 7, "008": 7, "002": 7, "005": 7, "004": 7, "001": 7, "_n": [8, 9, 10, 11, 12, 14, 15, 18, 22], "n_1": 8, "n_2": 8, "bar": [8, 13], "lvert": [8, 9, 11, 13, 15, 18, 20, 21, 22], "rvert": [8, 9, 15, 18, 22], "sigma_1": [8, 9], "sigma_2": [8, 9], "mu_1": 8, "mu_2": 8, "s_w": 8, "s_1": [8, 14], "s_2": [8, 14], "nu": [8, 20, 21], "equat": [9, 20, 21], "a_": [9, 11, 12, 13, 14], "x_ix_j": 9, "ii": [9, 18], "ji": [9, 11, 18], "succ": [9, 20, 21], "succeq": [9, 15, 20, 21], "prec": [9, 20], "preceq": [9, 20], "tr": [9, 11, 12, 18], "det": [9, 13], "cccc": 9, "12": [9, 12, 22], "1n": [9, 11, 12], "21": [9, 12], "22": [9, 12], "n1": 9, "n2": 9, "nn": [9, 11], "i1": [9, 18], "1j": 9, "nj": [9, 18], "prod_i": 9, "leftrightarrow": [9, 11, 15, 18, 21], "cc": [9, 11, 12, 22], "schwartz": [9, 15], "fischer": 9, "minkowski": 9, "vec": [9, 11, 12], "l_0": [9, 21], "rvert_0": 9, "l_1": 9, "rvert_1": 9, "dot": [9, 10, 18, 22], "rvert_2": [9, 11, 20, 21, 22], "l_": [9, 20], "rvert_": 9, "rvert_p": 9, "max_": [9, 18], "nx_j": 9, "sigma_": 9, "frobenius": 9, "nuclear": 9, "min": [9, 18, 19, 20, 21, 22], "sigma_r": 9, "svd": 9, "sigma": [9, 10, 11, 14, 18, 19], "uv": [9, 11], "neurip": 9, "factor": 9, "group": 9, "spars": 9, "regular": 9, "effici": 9, "low": [9, 18], "rank": [9, 10, 12, 13, 14, 18], "matrix": [9, 17, 18], "recoveri": 9, "pdf": 9, "sp": 9, "pm": [9, 11, 12], "abc": [9, 11], "bca": [9, 11], "cab": [9, 11], "yx": 9, "lambda_1": [9, 10, 18], "lambda_2": [9, 10, 18], "lambda_n": [9, 10], "lambda_i": [9, 10, 18, 20, 21], "4082": 10, "1826": 10, "8944": 10, "8165": 10, "3651": 10, "4472": 10, "9129": 10, "0000": [10, 13], "4495": 10, "4907": 10, "5320": 10, "8257": 10, "q1": 10, "r1": 10, "2673": 10, "8729": 10, "5345": 10, "2182": 10, "8018": 10, "4364": 10, "4833": 10, "6036": 10, "7417": 10, "6547": 10, "ans": 10, "ddot": [10, 11], "sigma_i": [10, 14, 18], "_m": [10, 11, 12, 14, 18], "av": 10, "aa": 10, "x_m": [11, 12, 15, 18], "ccc": 11, "m1": [11, 12], "nabla_": [11, 20, 21], "c_1": [11, 15], "c_2": [11, 15], "c_1f": 11, "c_2g": 11, "dg": 11, "ll": [11, 18, 21], "ki": 11, "lj": 11, "wedg": [11, 20], "x_kx_l": 11, "na_": 11, "il": [11, 18], "x_l": 11, "x_k": 11, "top_i": 11, "axb": 11, "uvw": 11, "vw": 11, "otim": [11, 12], "top_": 11, "jacobin": 11, "b_": [11, 12, 13, 18], "xw": 11, "softmax": 11, "wx": [11, 19], "hot": 11, "ccccccc": 11, "p1": [11, 12], "1q": [11, 12], "pq": [11, 12, 18], "bk": 11, "2f": [11, 18, 22], "nm": 11, "oplus": [12, 14], "ac": 12, "bd": 12, "m2": 12, "mp": 12, "nq": 12, "2q": 12, "p2": 12, "a_1b_1": 12, "a_1b_2": 12, "a_mb_p": 12, "cd": 12, "ace": 12, "bdf": 12, "ai": 12, "xb": 12, "dagger": 12, "bi": [13, 18], "hermitain": 13, "0000i": 13, "uu": 13, "hu": 13, "rvert_f": 13, "ba": 13, "qq": 13, "span": 14, "a_1": 14, "a_m": 14, "s_n": 14, "cup": 14, "_s": 14, "_h": 14, "rang": [14, 18], "col": [14, 18], "row": [14, 17, 18], "null": 14, "ker": 14, "dim": 14, "_k": [14, 18, 20, 22], "_r": 14, "ct": 15, "c_1t": 15, "c_2t": 15, "90": 15, "g_": 15, "x_iy_j": 15, "vector": 15, "frac14": 15, "leq": [15, 18, 20, 22], "v_i": [15, 18], "41": 15, "414": 15, "4142": 15, "notin": [15, 18], "complet": [15, 16], "vert_2": [15, 19, 20], "expect": 16, "maximinz": 16, "algorithm": 16, "ell": 16, "data": [16, 17, 18], "likelihood": 16, "ell_c": 16, "auxiliari": 16, "arg": [16, 20, 22], "sum_zp": 16, "z_j": [16, 18], "mu_j": 16, "author": 16, "jimilaw": 16, "geti": 16, "estep": 16, "numer": 16, "power": 16, "denomin": 16, "max_iter": 16, "1000": 16, "iter": 16, "init": 16, "param": 16, "while": [16, 22], "calcul": 16, "updat": 16, "tp": 16, "tq": 16, "given": 17, "column": 17, "block": 17, "cluster": 17, "co": 17, "or": 17, "mode": 17, "abl": 17, "simultan": 17, "exhibit": 17, "behavior": 17, "across": 17, "vice": 17, "versa": 17, "princip": 18, "compon": 18, "analysi": 18, "pca": 18, "_d": 18, "wz": 18, "parallel": 18, "z_": 18, "i2": 18, "id": 18, "const": 18, "text": 18, "propto": 18, "xx": 18, "w_1": 18, "w_2": 18, "w_": 18, "matplotlib": 18, "pyplot": 18, "plt": 18, "n_sampl": 18, "n_featur": 18, "scatter_matrix": 18, "transpos": 18, "eig_val": 18, "eig_vec": 18, "linalg": [18, 22], "eig": 18, "eig_pair": 18, "abs": [18, 22], "sort": 18, "revers": 18, "true": [18, 22], "ele": 18, "x_new": 18, "plot": 18, "ro": 18, "marker": 18, "linspac": 18, "x1": 18, "arctan": 18, "y1": 18, "proj_dir": 18, "append": 18, "ix": 18, "scatter": 18, "zip": 18, "annot": 18, "0f": 18, "xytext": 18, "textcoord": 18, "offset": 18, "equal": 18, "pca_svd": 18, "vt": 18, "x_new_svd": 18, "f_svd": 18, "x_eig_decom": 18, "x_svd": 18, "xu": 18, "lambda_j": 18, "gram": 18, "mercer": 18, "kpca": 18, "ko": 18, "ok": 18, "oko": 18, "paramet": 18, "dimens": 18, "project": 18, "ndimens": 18, "output": 18, "lesser": 18, "than": 18, "input": 18, "construct": 18, "zero": 18, "k_ij": 18, "all1": 18, "k_center": 18, "eigvector": 18, "diag": [18, 20], "multipl": 18, "dimension": 18, "scale": 18, "mds": 18, "dist_": 18, "z_i": 18, "2z_i": 18, "tz_j": 18, "jj": 18, "2b_": 18, "mb_": 18, "mdist_": 18, "2m": 18, "center": 18, "sklearn": 18, "dataset": 18, "manifold": 18, "ris": 18, "load_iri": 18, "iri": 18, "target": 18, "subplot": 18, "121": 18, "n_compon": 18, "fit": 18, "new_x_pca": 18, "transform": 18, "122": 18, "metric": 18, "new_x_md": 18, "fit_transform": 18, "show": 18, "isometr": 18, "isomap": 18, "2d": 18, "dijkstra": 18, "floyd": 18, "dist": 18, "nearest": 18, "neighbor": 18, "otherwis": 18, "fig": 18, "figsiz": 18, "15": [18, 22], "idx": 18, "enumer": 18, "20": 18, "100": 18, "n_neighbor": 18, "set_titl": 18, "emptyset": 18, "homeomorph": 18, "isomorph": 18, "continu": 18, "_l": 18, "ik": 18, "step": 18, "min_w": 18, "varepsilon": 18, "varepsilon_i": 18, "weight": 18, "jk": 18, "ls": 18, "vert_f": 18, "min_": 18, "zmz": 18, "zz": 18, "div": 18, "grad": 18, "v_j": 18, "laplacian": 18, "f_x": 18, "4f": 18, "wf": 18, "df": 18, "2w_": 18, "_iw_": 18, "_jw_": 18, "ly": 18, "stochast": 18, "q_": 18, "nl": 18, "sum_k": 18, "sum_l": 18, "s_i": 18, "101": 18, "visual": 18, "hz": 19, "mu_k": [20, 22], "h_1": 20, "h_q": 20, "h_i": [20, 21], "rho_k": 20, "bz": 20, "kkt": 20, "i_": 20, "tf_0": 20, "central": 20, "path": 20, "lambda_if_i": [20, 21], "tf_i": 20, "color": 20, "red": 20, "nt": [20, 22], "2f_0": 20, "nu_": 20, "r_t": 20, "f_m": [20, 21], "r_": 20, "pri": 20, "cent": 20, "succ0": 20, "dr_t": 20, "2f_i": 20, "y_": 20, "pd": 20, "lambda_k": 20, "nu_k": 20, "not": 20, "necessar": 20, "feasitbl": 20, "eta_k": 20, "surrog": 20, "dualiti": 20, "gap": 20, "epsilon_": 20, "fea": 20, "until": 20, "af_1": 21, "bf_2": 21, "f_a": 21, "leq0": 21, "nu_ih_i": 21, "nu_i": 21, "j_1": 21, "j_p": 21, "j_i": 21, "supremum": 21, "j_2": 21, "j_d": 21, "succeq0": [21, 22], "le0": 21, "ge0": [21, 22], "relint": 21, "opt": 22, "alpha_k": 22, "_0": 22, "ad": 22, "alpha_0": 22, "alpha_": 22, "13": 22, "alpha_n": 22, "beta_k": 22, "14": 22, "beta_i": 22, "linear_conj_desc": 22, "descent": 22, "direct": 22, "dtype": 22, "float": 22, "nf_": 22, "7763568394002505e": 22, "alpha_j": 22, "beta_j": 22, "dom": 22, "67": 22, "nablaf": 22, "nabla2f": 22, "10000": 22, "1e": 22, "lambda2": 22, "newton_step": 22, "tf": 22, "tlambda2": 22, "cx": 22, "includ": 22, "iostream": 22, "cmath": 22, "namespac": 22, "std": 22, "doubl": 22, "main": 22, "cout": 22, "cin": 22, "endl": 22, "pow": 22, "cnt": 22, "tx": 22, "break": 22}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"indic": 0, "and": 0, "tabl": 0, "mean": 1, "embed": 1, "maximum": 1, "discrep": 1, "integr": 1, "probabl": 1, "metric": 1, "ipm": 1, "mmd": 1, "hilbert": [2, 4], "rkhs": 2, "schmidt": [4, 10], "rank": 4, "one": 4, "cross": 4, "covari": 4, "otim": 4, "coco": 4, "bregman": 5, "diverg": 5, "squar": 5, "euclidean": 5, "distanc": 5, "refer": 5, "gamma": 6, "beta": 6, "binomi": 6, "python": 7, "mu": 8, "sigma": 8, "schatten": 9, "qr": 10, "gram": 10, "svd": [10, 18], "jacobian": 11, "mathbf": [11, 22], "hessian": 11, "hadamard": 12, "kroneck": 12, "hermitian": 13, "em": 16, "biclust": 17, "biclustr": 17, "via": 17, "singlular": 17, "valu": 17, "decomposit": 17, "lle": 18, "sne": 18, "autoencod": 19, "lagrangian": [20, 21], "alm": 20, "admm": 20, "kkt": 21, "slater": 21, "newton": 22}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 60}, "alltitles": {"\u673a\u5668\u5b66\u4e60\u57fa\u7840": [[0, "id1"]], "\u76d1\u7763\u5b66\u4e60": [[0, null]], "\u65e0\u76d1\u7763\u5b66\u4e60": [[0, null]], "\u6838\u65b9\u6cd5": [[0, null]], "\u6700\u4f18\u5316": [[0, null]], "\u77e9\u9635\u5206\u6790": [[0, null]], "\u6570\u5b66\u5efa\u6a21": [[0, null]], "Indices and tables": [[0, "indices-and-tables"]], "\u6700\u5927\u5747\u503c\u5dee": [[1, "id1"]], "\u5747\u503c\u5d4c\u5165 (Mean Embedding)": [[1, "mean-embedding"]], "\u6700\u5927\u5747\u503c\u5dee\u5206(Maximum Mean Discrepancy)": [[1, "maximum-mean-discrepancy"]], "\u4e00\u79cd\u79ef\u5206\u6982\u7387\u5ea6\u91cf": [[1, "id2"]], "\u79ef\u5206\u6982\u7387\u5ea6\u91cf(Integral Probability Metric, IPM)": [[1, "integral-probability-metric-ipm"]], "MMD\u79ef\u5206\u6982\u7387\u5ea6\u91cf": [[1, "mmd"]], "\u518d\u751f\u6838\u5e0c\u5c14\u4f2f\u7279\u7a7a\u95f4": [[2, "id1"]], "\u57fa\u672c\u6982\u5ff5": [[2, "id2"]], "\u6cdb\u51fd": [[2, "id3"]], "\u6cdb\u51fd\u6781\u503c\u6c42\u89e3\u2013\u53d8\u5206\u6cd5": [[2, "id4"]], "\u542b\u53c2\u53d8\u91cf\u7684\u5b9a\u79ef\u5206": [[2, "id5"]], "\u4f8b\u5b50": [[2, "id6"]], "\u7ebf\u6027\u7b97\u5b50": [[2, "id7"]], "\u518d\u751f\u6838Hilbert\u7a7a\u95f4": [[2, "hilbert"]], "\u6c42\u503c\u6cdb\u51fd": [[2, "id8"]], "\u518d\u751f\u6838": [[2, "id9"]], "RKHS\u5177\u4f53\u5316": [[2, "rkhs"]], "\u6838\u51fd\u6570\u57fa\u7840": [[3, "id1"]], "\u6838\u51fd\u6570": [[3, "id2"]], "\u793a\u4f8b": [[3, "id3"]], "\u4ec0\u4e48\u662f\u6838\u51fd\u6570": [[3, "id4"]], "\u5e38\u7528\u6838\u51fd\u6570": [[3, "id5"]], "\u7279\u5f81\u6620\u5c04\u7684\u57fa\u672c\u8fd0\u7b97": [[3, "id6"]], "\u6295\u5f71": [[3, "id7"]], "\u534f\u65b9\u5dee\u7b97\u5b50": [[4, "id1"]], "Hilbert-Schmidt\u7b97\u5b50": [[4, "hilbert-schmidt"]], "Rank-one\u7b97\u5b50": [[4, "rank-one"]], "Cross-covariance\u7b97\u5b50": [[4, "cross-covariance"]], "Cross-covariance\u7b97\u5b50\u5b9a\u4e49": [[4, "id2"]], "\u7279\u4f8b\u2014\u2014f\\otimes g": [[4, "f-otimes-g"]], "\u7ea6\u675f\u534f\u65b9\u5dee": [[4, "id3"]], "COCO\u7684\u4f30\u8ba1": [[4, "coco"]], "Bregman divergence": [[5, "bregman-divergence"]], "Squared Euclidean Distance": [[5, "squared-euclidean-distance"]], "Bregman divergences": [[5, "bregman-divergences"]], "Reference": [[5, "reference"]], "\u5171\u8f6d\u5206\u5e03": [[6, "id1"]], "Gamma\u5206\u5e03": [[6, "gamma"], [6, "id3"]], "Gamma\u51fd\u6570": [[6, "id2"]], "Beta\u5206\u5e03": [[6, "beta"]], "Beta-Binomial\u5171\u8f6d": [[6, "beta-binomial"]], "\u4fe1\u606f\u71b5": [[7, "id1"], [7, "id3"]], "\u4fe1\u606f\u91cf": [[7, "id2"]], "Python\u4ee3\u7801": [[7, "python"]], "\u4e0d\u786e\u5b9a\u6a21\u578b": [[8, "id1"]], "\u62bd\u6837\u5206\u5e03": [[8, "id2"]], "\u53c2\u6570\u533a\u95f4\u4f30\u8ba1\u4e0e\u5047\u8bbe\u68c0\u9a8c": [[8, "id3"]], "\\mu\u5747\u503c\u68c0\u6d4b": [[8, "mu"]], "\u5355\u4e2a\u603b\u4f53": [[8, "id4"]], "\u4e24\u4e2a\u603b\u4f53": [[8, "id5"]], "\\sigma\u65b9\u5dee\u6bd4\u68c0\u6d4b": [[8, "sigma"]], "\u77e9\u9635\u6027\u80fd\u6307\u6807": [[9, "id1"]], "\u4e8c\u6b21\u578b": [[9, "id2"]], "\u5e38\u7528\u6027\u8d28": [[9, "id3"]], "\u884c\u5217\u5f0f": [[9, "id4"]], "\u5b9a\u4e49": [[9, "id5"]], "\u4f59\u5b50\u5f0f": [[9, "id6"]], "\u6027\u8d28\u4e00": [[9, "id7"], [9, "id18"]], "\u6027\u8d28\u4e8c": [[9, "id8"], [9, "id19"]], "\u77e9\u9635\u5185\u79ef": [[9, "id9"]], "\u77e9\u9635\u8303\u6570": [[9, "id10"], [9, "id12"]], "\u5411\u91cf\u8303\u6570": [[9, "id11"]], "\u8bf1\u5bfc\u8303\u6570": [[9, "id13"]], "\u5143\u7d20\u5f62\u5f0f\u8303\u6570": [[9, "id14"]], "Schatten\u8303\u6570": [[9, "schatten"], [9, "id16"]], "\u6838\u8303\u6570": [[9, "id15"]], "\u8ff9": [[9, "id17"]], "\u77e9\u9635\u5206\u89e3": [[10, "id1"]], "QR\u5206\u89e3": [[10, "qr"]], "QR\u7684\u4e00\u822c\u5f62\u5f0f": [[10, "id2"]], "QR\u5206\u89e3\u7684Gram-Schmidt\u6b63\u4ea4\u5316\u65b9\u6cd5": [[10, "qrgram-schmidt"]], "\u7279\u5f81\u503c\u5206\u89e3": [[10, "id3"]], "\u65b9\u9635\u7684\u7279\u5f81\u503c\u5206\u89e3": [[10, "id4"]], "\u5bf9\u79f0\u77e9\u9635\u7684\u7279\u5f81\u503c\u5206\u89e3": [[10, "id5"]], "\u5947\u5f02\u503c\u5206\u89e3": [[10, "id6"]], "SVD\u53c2\u6570\u6c42\u89e3": [[10, "svd"]], "\u77e9\u9635\u5fae\u5206": [[11, "id1"]], "Jacobian\u77e9\u9635": [[11, "jacobian"]], "\u68af\u5ea6\u77e9\u9635": [[11, "id2"]], "\u504f\u5bfc\u548c\u68af\u5ea6\u8ba1\u7b97": [[11, "id3"]], "\u57fa\u672c\u89c4\u5219": [[11, "id4"]], "\u72ec\u7acb\u6027\u5047\u8bbe": [[11, "id5"]], "\u6848\u4f8b": [[11, "id6"], [11, "id11"], [16, "id3"], [22, "id7"]], "\u4e00\u9636\u5b9e\u77e9\u9635\u5fae\u5206": [[11, "id7"]], "\u6027\u8d28": [[11, "id8"]], "\u6807\u91cf\u51fd\u6570f(\\mathbf{x})\u7684\u5411\u91cf\u53d8\u5143\\mathbf{x}\u5168\u5fae\u5206\u6c42\u504f\u5bfc\u65b9\u6cd5": [[11, "f-mathbf-x-mathbf-x"]], "\u6807\u91cf\u51fd\u6570f(\\mathbf{X})\u7684\u77e9\u9635\u91cf\u53d8\u5143\\mathbf{X}\u5168\u5fae\u5206\u6c42\u504f\u5bfc\u65b9\u6cd5": [[11, "id9"]], "\u6c42\u5bfc\u65b9\u6cd5": [[11, "id10"]], "\u5b9e\u77e9\u9635\u51fd\u6570\u7684\u504f\u5bfc\u8ba1\u7b97": [[11, "id12"]], "Hessian\u77e9\u9635": [[11, "hessian"]], "\u77e9\u9635\u8fd0\u7b97": [[12, "id1"]], "\u76f4\u548c": [[12, "id2"]], "Hadamard\u79ef": [[12, "hadamard"]], "Kronecker\u79ef": [[12, "kronecker"]], "\u5411\u91cf\u5316": [[12, "id3"]], "\u7279\u6b8a\u77e9\u9635": [[13, "id1"]], "Hermitian\u77e9\u9635": [[13, "hermitian"]], "\u9149\u77e9\u9635": [[13, "id2"]], "\u6b63\u4ea4\u77e9\u9635": [[13, "id3"]], "\u5b50\u7a7a\u95f4\u5206\u6790": [[14, "id1"]], "\u4ec0\u4e48\u662f\u5b50\u7a7a\u95f4": [[14, "id2"]], "\u6982\u5ff5": [[14, "id3"], [15, "id2"]], "\u6b63\u4ea4\u8865": [[14, "id4"]], "\u6b63\u4ea4\u6295\u5f71": [[14, "id5"]], "\u5217\uff08\u884c\uff09\u7a7a\u95f4\u4e0e\u96f6\u7a7a\u95f4": [[14, "id6"]], "\u5b50\u7a7a\u95f4\u57fa\u7684\u6784\u9020": [[14, "id7"]], "\u7a7a\u95f4\u57fa\u6784\u9020\u7684\u5947\u5f02\u503c\u5206\u89e3\u65b9\u6cd5": [[14, "id8"]], "\u5411\u91cf\u7a7a\u95f4": [[15, "id1"], [15, "id3"]], "\u5185\u79ef\u7a7a\u95f4": [[15, "id4"]], "\u8d4b\u8303\u7a7a\u95f4": [[15, "id5"]], "EM\u7b97\u6cd5\u6982\u8ff0": [[16, "em"]], "EM\u7b97\u6cd5": [[16, "id1"]], "EM\u7b97\u6cd5\u63a8\u5bfc": [[16, "id2"]], "\u5b9e\u9a8c\u4ee3\u7801": [[16, "id4"]], "BiClustering": [[17, "biclustering"]], "Biclustring via singlular value decomposition": [[17, "biclustring-via-singlular-value-decomposition"]], "\u6570\u636e\u964d\u7ef4": [[18, "id1"]], "\u4e3b\u6210\u5206\u5206\u6790": [[18, "id2"]], "\u6700\u8fd1\u91cd\u6784\u6027": [[18, "id3"]], "\u6700\u5927\u53ef\u5206\u6027": [[18, "id4"]], "\u4f18\u5316\u95ee\u9898\u7684\u6c42\u89e3": [[18, "id5"]], "\u7b97\u6cd51\u2013\u7279\u5f81\u503c\u5206\u89e3": [[18, "id6"]], "\u7b97\u6cd52\u2013SVD\u5206\u89e3": [[18, "svd"]], "\u6838\u4e3b\u6210\u5206\u5206\u6790": [[18, "id7"]], "\u591a\u7ef4\u7f29\u653e": [[18, "id8"]], "\u7b97\u6cd5": [[18, "id9"], [18, "id11"]], "\u7b49\u5ea6\u91cf\u6620\u5c04": [[18, "id10"]], "\u6d41\u5f62": [[18, "id12"]], "LLE\u5c40\u90e8\u7ebf\u6027\u5d4c\u5165": [[18, "lle"]], "LLE\u57fa\u672c\u601d\u60f3": [[18, "id13"]], "LLE\u6c42\u89e3": [[18, "id14"]], "\u62c9\u666e\u62c9\u65af\u7279\u5f81\u6620\u5c04": [[18, "id15"]], "\u62c9\u666e\u62c9\u65af\u7b97\u5b50": [[18, "id16"]], "\u62c9\u666e\u62c9\u65af\u77e9\u9635": [[18, "id17"]], "\u62c9\u666e\u62c9\u65af\u53d8\u6362": [[18, "id18"]], "\u968f\u673a\u8fd1\u90bb\u5d4c\u5165": [[18, "id19"]], "SNE": [[18, "sne"]], "\u5bf9\u79f0SNE": [[18, "id20"]], "t-SNE": [[18, "t-sne"]], "\u8868\u793a\u5b66\u4e60": [[19, "id1"]], "\u81ea\u7f16\u7801\u5668(AutoEncoder)": [[19, "autoencoder"]], "\u4f18\u5316\u95ee\u9898\u6c42\u89e3(2)": [[20, "id1"]], "Lagrangian\u4e58\u5b50\u6cd5": [[20, "lagrangian"], [21, "lagrangian"]], "\u589e\u5e7fLagrangian\u4e58\u5b50\u6cd5(ALM)": [[20, "lagrangian-alm"]], "\u7b49\u5f0f\u7ea6\u675f": [[20, "id2"]], "\u6df7\u5408\u7ea6\u675f": [[20, "id3"]], "\u4ea4\u66ff\u65b9\u5411\u4e58\u5b50\u6cd5(ADMM)": [[20, "admm"]], "\u5185\u70b9\u6cd5": [[20, "id4"]], "\u969c\u788d\u6cd5": [[20, "id5"]], "\u539f\u5bf9\u5076\u5185\u70b9\u6cd5": [[20, "id6"]], "\u539f\u5bf9\u5076\u641c\u7d22\u65b9\u5411": [[20, "id7"]], "\u4ee3\u7406\u5bf9\u5076\u95f4\u9699": [[20, "id8"]], "\u51f8\u4f18\u5316\u95ee\u9898": [[21, "id1"], [21, "id5"]], "\u51f8\u51fd\u6570": [[21, "id2"]], "\u5224\u5b9a\u65b9\u6cd5": [[21, "id3"]], "\u4fdd\u51f8\u8fd0\u7b97\u4e0e\u6027\u8d28\u5224\u5b9a": [[21, "id4"]], "\u5bf9\u5076\u65b9\u6cd5": [[21, "id6"]], "KKT\u6761\u4ef6": [[21, "kkt"]], "\u5f3a\u5bf9\u5076\u6027\uff08Slater\u5b9a\u7406\uff09": [[21, "slater"]], "\u4f18\u5316\u95ee\u9898\u6c42\u89e3(1)": [[22, "id1"]], "\u4e0b\u964d\u6cd5": [[22, "id2"]], "\u6700\u901f\u4e0b\u964d\u6cd5": [[22, "id3"]], "Newton\u6cd5": [[22, "newton"]], "\u68af\u5ea6\u6295\u5f71\u6cd5": [[22, "id4"]], "\u5171\u8f6d\u68af\u5ea6\u4e0b\u964d\u6cd5": [[22, "id5"]], "\\mathbf{A}-\u5171\u8f6d": [[22, "mathbf-a"]], "\u5171\u8f6d\u68af\u5ea6\u6cd5": [[22, "id6"]], "\u4e00\u822c\u51fd\u6570\u7684\u5171\u8f6d\u68af\u5ea6\u6cd5": [[22, "id8"]], "Newton\u65b9\u6cd5": [[22, "id9"]]}, "indexentries": {}})