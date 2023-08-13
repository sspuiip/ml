Search.setIndex({"docnames": ["index", "kernel/MMD", "kernel/RKHS", "kernel/base", "mathmodel/statistic_model", "matrix/base", "matrix/decomposition", "matrix/matrixdiff", "matrix/matrixoper", "matrix/special_matrix", "matrix/subspace", "matrix/vectorspace", "ml/dimension_reduce", "ml/neuro_representation", "optimization/convex_neq_solve", "optimization/convex_prob", "optimization/convex_solve"], "filenames": ["index.rst", "kernel/MMD.md", "kernel/RKHS.md", "kernel/base.md", "mathmodel/statistic_model.md", "matrix/base.md", "matrix/decomposition.md", "matrix/matrixdiff.md", "matrix/matrixoper.md", "matrix/special_matrix.md", "matrix/subspace.md", "matrix/vectorspace.md", "ml/dimension_reduce.md", "ml/neuro_representation.md", "optimization/convex_neq_solve.md", "optimization/convex_prob.md", "optimization/convex_solve.md"], "titles": ["\u673a\u5668\u5b66\u4e60\u57fa\u7840", "<span class=\"section-number\">3. </span>\u6700\u5927\u5747\u503c\u5dee (Maximum Mean Discrepancy)", "<span class=\"section-number\">2. </span>\u518d\u751f\u6838\u5e0c\u5c14\u4f2f\u7279\u7a7a\u95f4", "<span class=\"section-number\">1. </span>\u6838\u51fd\u6570\u57fa\u7840", "<span class=\"section-number\">1. </span>\u4e0d\u786e\u5b9a\u6a21\u578b", "<span class=\"section-number\">1. </span>\u77e9\u9635\u6027\u80fd\u6307\u6807", "<span class=\"section-number\">7. </span>\u77e9\u9635\u5206\u89e3", "<span class=\"section-number\">4. </span>\u77e9\u9635\u5fae\u5206", "<span class=\"section-number\">2. </span>\u77e9\u9635\u8fd0\u7b97", "<span class=\"section-number\">5. </span>\u7279\u6b8a\u77e9\u9635", "<span class=\"section-number\">6. </span>\u5b50\u7a7a\u95f4\u5206\u6790", "<span class=\"section-number\">3. </span>\u5411\u91cf\u7a7a\u95f4", "<span class=\"section-number\">1. </span>\u6570\u636e\u964d\u7ef4", "<span class=\"section-number\">2. </span>\u8868\u793a\u5b66\u4e60", "<span class=\"section-number\">3. </span>\u4f18\u5316\u95ee\u9898\u6c42\u89e3(2)", "<span class=\"section-number\">1. </span>\u51f8\u4f18\u5316\u95ee\u9898", "<span class=\"section-number\">2. </span>\u4f18\u5316\u95ee\u9898\u6c42\u89e3(1)"], "terms": {"lle": 0, "autoencod": 0, "hilbert": [0, 1, 3, 11], "maximum": 0, "mean": [0, 12], "discrep": 0, "embed": [0, 12], "newton": [0, 14], "lagrangian": [0, 12, 16], "alm": 0, "admm": 0, "hadamard": [0, 5, 7], "kroneck": [0, 7], "jacobian": 0, "hessian": [0, 14, 16], "hermitian": [0, 10], "qr": [0, 10], "mathcal": [1, 2, 3, 4, 11, 12, 13, 15, 16], "borel": 1, "mu_p": 1, "mathbb": [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "_p": [1, 8], "varphi_i": 1, "sim": [1, 4, 13], "langl": [1, 2, 3, 5, 6, 11, 12, 14, 15], "mu_q": 1, "rangle_": [1, 2, 3], "varphi": 1, "cdot": [1, 2, 3, 5, 6, 7, 8, 11, 12, 14, 15, 16], "begin": [1, 2, 3, 5, 6, 7, 8, 10, 12, 14, 15, 16], "bmatrix": [1, 6, 8, 10, 12, 14], "end": [1, 2, 3, 5, 6, 7, 8, 10, 12, 14, 15, 16], "quad": [1, 2, 3, 4, 5, 6, 10, 12, 13, 14, 15, 16], "top": [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16], "ax": [1, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16], "bx": [1, 14, 16], "left": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16], "right": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16], "triangleq": [1, 3, 5, 7, 12, 13, 16], "varphi_1": 1, "varphi_2": 1, "in": [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "sqrt": [1, 2, 3, 4, 5, 6, 11, 12], "infti": [1, 2, 3, 5, 11, 12, 13, 14, 15], "exist": [1, 11], "t_pf": 1, "foral": [1, 2, 3, 5, 6, 9, 10, 11, 12, 15], "split": [1, 2, 3, 5, 6, 7, 10, 12, 14, 15, 16], "le": [1, 2, 3, 4, 5, 6, 9, 11, 12, 14, 15, 16], "vert": [1, 2, 3, 6, 11, 12, 13, 14], "vert_": [1, 2, 11], "riesz": [1, 2], "lambda_": [1, 5, 12, 14], "t_p": 1, "g_a": 1, "af": [1, 2], "lambda_a": 1, "textrm": [1, 2, 3, 5, 12, 14], "mmd": 1, "underbrac": [1, 2, 3, 12], "_q": [1, 8], "within": 1, "distrib": 1, "similar": 1, "cross": 1, "rangl": [1, 2, 3, 5, 6, 11, 12, 14, 15], "integr": 1, "probabl": 1, "metric": [1, 12], "ipm": 1, "d_": [1, 7, 12, 13], "sup_": [1, 2], "name": 1, "formula": 1, "condit": [1, 3], "dudley": 1, "bl": 1, "vert_l": 1, "sup": [1, 2, 14, 15], "rho": [1, 14], "neq": [1, 4, 5, 6, 7, 8, 12, 16], "wasserstein": 1, "inf_": 1, "mu": [1, 11, 14, 15], "int": [1, 2, 12, 16], "and": [1, 3, 14], "is": [1, 2, 3, 12], "separ": 1, "rkhs": 1, "frac": [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16], "box": [1, 2, 12], "approx": [1, 12, 14, 16], "frac1m": 1, "sum_": [1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 14, 15, 16], "x_i": [1, 2, 5, 7, 12], "frac1n": [1, 4, 12], "x_j": [1, 2, 5, 7, 10], "x_": [1, 7, 9, 12, 14, 16], "mn": [1, 5, 7, 8, 12], "function": [2, 12], "rightarrow": [2, 3, 7, 8, 10, 11, 12, 13, 14, 15, 16], "subseteq": [2, 11, 12], "int_": [2, 11], "x_0": 2, "x_1": [2, 3, 5, 7, 8, 12, 16], "dx": [2, 11], "ds": 2, "dy": [2, 7, 12], "int_0": 2, "mgi": 2, "mv": 2, "2gi": 2, "dt": 2, "hat": [2, 3, 8, 12, 13, 14], "calculus": 2, "of": [2, 3, 12], "variat": 2, "epsilon": [2, 11, 14, 15, 16], "eta": [2, 14], "delta": [2, 12, 14, 16], "partial": [2, 5, 7, 12, 15, 16], "boundari": 2, "inf": [2, 14], "qquad": [2, 12, 15], "euler": 2, "lagrang": [2, 14], "int_a": 2, "f_i": [2, 12, 14, 15], "_y": 2, "isbn": 2, "9787040427806": 2, "kx": 2, "frac12": [2, 5, 7, 12, 14, 15, 16], "2r": 2, "sin": [2, 12], "theta": [2, 13, 15, 16], "cos": [2, 12, 16], "array": [2, 5, 6, 7, 8, 12, 14, 15, 16], "homogen": 2, "alpha": [2, 3, 7, 16], "addit": [2, 3], "ag": 2, "subset": 2, "banach": [2, 11], "l_p": [2, 3, 5], "a_g": 2, "norm": [2, 5, 11, 12, 16], "linear": 2, "space": [2, 3, 10, 11], "over": 2, "itself": 2, "f_0": [2, 14, 15], "impli": 2, "af_0": 2, "f_1": [2, 3, 14, 15], "f_2": [2, 3, 15], "oper": 2, "bound": 2, "lambda": [2, 5, 6, 12, 14, 15, 16], "one": [2, 7, 12], "point": [2, 12, 14], "algebra": 2, "dual": [2, 14], "topolog": [2, 12], "represent": 2, "for": [2, 3, 5, 12], "some": 2, "dirac": 2, "evalu": 2, "let": 2, "be": [2, 3, 12], "defin": [2, 3], "on": [2, 3], "non": 2, "empti": 2, "set": 2, "fix": 2, "map": [2, 3, 12], "delta_x": 2, "call": 2, "the": [2, 3, 12], "at": 2, "beta": [2, 7, 16], "reproduc": 2, "kernel": [2, 3, 10, 12], "coroallari": 2, "lim_": [2, 11], "f_n": 2, "l_2": [2, 5], "q_n": 2, "2n": [2, 5, 8, 12], "kappa": [2, 3], "time": [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13], "properti": 2, "cauchi": [2, 3, 5, 11], "schwarz": [2, 3], "inequ": 2, "f_": [2, 7, 12, 16], "delta_": [2, 7, 13], "anti": 2, "isometri": 2, "ge": [2, 3, 5, 6, 11, 15], "a_i": [2, 3], "a_ia_jh": 2, "a_j": 2, "stricklli": 2, "posit": 2, "definit": 2, "lemma": [2, 3], "phi": [2, 3, 12, 13, 14], "corollari": 2, "moor": 2, "aronszajn": 2, "xor": 3, "pmb": [3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15], "x_2": [3, 5, 8, 12, 16], "x_1x_2": 3, "z_1": 3, "z_1z_2": 3, "z_2": 3, "2z_1": 3, "2z_2": 3, "2x_1x_2z_1z_2": 3, "x_1z_1": 3, "x_2z_2": 3, "trick": 3, "inner": [3, 11], "product": [3, 11], "alpha_1": [3, 16], "alpha_2": 3, "if": [3, 12, 16], "onli": 3, "hilber": [3, 11], "which": 3, "an": 3, "along": 3, "with": [3, 12], "technic": 3, "sum": [3, 12], "are": 3, "k_1": 3, "k_2": 3, "between": 3, "tild": [3, 4, 10, 12], "_1": [3, 4, 5, 6, 7, 8, 10, 12, 16], "_2": [3, 5, 6, 12, 16], "polynomi": 3, "ell_p": 3, "ell_2": 3, "phi_i": 3, "th": 3, "taylor": [3, 16], "seri": 3, "expans": 3, "may": 3, "use": [3, 12, 16], "to": [3, 12], "that": 3, "have": 3, "infinityli": 3, "mani": 3, "featur": [3, 12], "a_n": 3, "all": 3, "exampl": 3, "exponenti": 3, "exp": [3, 7, 8, 12], "gaussian": 3, "gamma": 3, "normal": [3, 12], "2xx": 3, "la_i": 3, "_i": [3, 4, 5, 6, 7, 10, 12, 13, 16], "a_ia_j": 3, "_j": [3, 5, 6, 10, 12, 16], "phi_": 3, "frac1l": 3, "frac2l": 3, "k11": [3, 12], "11": [3, 5, 7, 8, 11, 12, 16], "p_": [3, 12], "bot": [3, 10], "ww": 3, "_n": [4, 5, 6, 7, 8, 10, 11, 12, 16], "chi": 4, "sum_i": [4, 5, 11, 12, 15], "n_1": 4, "n_2": 4, "bar": [4, 9], "lvert": [4, 5, 7, 9, 11, 12, 14, 15, 16], "rvert": [4, 5, 11, 12, 16], "t_": [4, 12], "sigma_1": [4, 5], "sigma_2": [4, 5], "mu_1": 4, "mu_2": 4, "s_w": 4, "s_1": [4, 10], "s_2": [4, 10], "nu": [4, 14, 15], "equat": [5, 14, 15], "sum_j": [5, 10, 11, 12], "a_": [5, 7, 8, 9, 10], "ij": [5, 7, 8, 9, 11, 12], "x_ix_j": 5, "ii": [5, 12], "ji": [5, 7, 12], "succ": [5, 14, 15], "succeq": [5, 11, 14, 15], "prec": [5, 14], "preceq": [5, 14], "tr": [5, 7, 8, 12], "det": [5, 9], "cccc": 5, "12": [5, 8, 16], "1n": [5, 7, 8], "21": [5, 8], "22": [5, 8], "vdot": [5, 6, 7, 8, 12, 14], "n1": 5, "n2": 5, "nn": [5, 7], "i1": [5, 12], "1j": 5, "nj": [5, 12], "ab": [5, 7, 8, 9, 11], "prod_i": 5, "leftrightarrow": [5, 7, 11, 12, 15], "cc": [5, 7, 8, 16], "schwartz": [5, 11], "prod_": 5, "fischer": 5, "minkowski": 5, "vec": [5, 7, 8], "l_0": [5, 15], "rvert_0": 5, "l_1": 5, "rvert_1": 5, "dot": [5, 6, 12, 16], "x_n": [5, 7, 11, 12], "rvert_2": [5, 7, 14, 15, 16], "l_": [5, 14], "rvert_": 5, "max": [5, 14, 15], "rvert_p": 5, "max_": [5, 12], "nx_j": 5, "sigma_": 5, "frobenius": 5, "nuclear": 5, "min": [5, 12, 13, 14, 15, 16], "sigma_r": 5, "svd": 5, "sigma": [5, 6, 7, 10, 12, 13], "mathrm": [5, 7, 8, 9, 10, 14, 15], "uv": [5, 7], "neurip": 5, "factor": 5, "group": 5, "spars": 5, "regular": 5, "effici": 5, "low": [5, 12], "rank": [5, 6, 8, 9, 10, 12], "matrix": [5, 12], "recoveri": 5, "pdf": 5, "code": 5, "sp": 5, "pm": [5, 7, 8], "abc": [5, 7], "bca": [5, 7], "cab": [5, 7], "yx": 5, "lambda_1": [5, 6, 12], "lambda_2": [5, 6, 12], "lambda_n": [5, 6], "lambda_i": [5, 6, 12, 14, 15], "4082": 6, "1826": 6, "8944": 6, "8165": 6, "3651": 6, "4472": 6, "9129": 6, "0000": [6, 9], "4495": 6, "4907": 6, "5320": 6, "8257": 6, "q1": 6, "r1": 6, "2673": 6, "8729": 6, "5345": 6, "2182": 6, "8018": 6, "4364": 6, "4833": 6, "6036": 6, "7417": 6, "6547": 6, "ans": 6, "ddot": [6, 7], "pmatrix": [6, 12], "sigma_i": [6, 10, 12], "_m": [6, 7, 8, 10, 12], "av": 6, "aa": 6, "x_m": [7, 8, 11, 12], "ccc": 7, "m1": [7, 8], "nabla_": [7, 14, 15], "c_1": [7, 11], "c_2": [7, 11], "c_1f": 7, "c_2g": 7, "dg": 7, "kl": [7, 12], "ll": [7, 12, 15], "ki": 7, "lj": 7, "wedg": [7, 14], "other": 7, "x_kx_l": 7, "na_": 7, "il": [7, 12], "x_l": 7, "x_k": 7, "top_i": 7, "axb": 7, "uvw": 7, "vw": 7, "log": [7, 12, 14], "otim": [7, 8], "top_": 7, "jacobin": 7, "b_": [7, 8, 9, 12], "c_": [7, 12], "xw": 7, "softmax": 7, "wx": [7, 13], "hot": 7, "ccccccc": 7, "p1": [7, 8], "1q": [7, 8], "pq": [7, 8, 12], "bk": 7, "2f": [7, 12, 16], "nm": 7, "oplus": [8, 10], "ac": 8, "bd": 8, "m2": 8, "mp": 8, "nq": 8, "2q": 8, "p2": 8, "a_1b_1": 8, "a_1b_2": 8, "a_mb_p": 8, "cd": 8, "ace": 8, "bdf": 8, "ai": 8, "xb": 8, "dagger": 8, "bi": [9, 12], "hermitain": 9, "0000i": 9, "uu": 9, "hu": 9, "rvert_f": 9, "ba": 9, "qq": 9, "span": 10, "a_1": 10, "a_m": 10, "s_n": 10, "cap": 10, "cup": 10, "_s": 10, "_h": 10, "px": [10, 12], "rang": [10, 12], "col": [10, 12], "row": [10, 12], "null": 10, "ker": 10, "dim": 10, "_k": [10, 12, 14, 16], "_r": 10, "ct": 11, "c_1t": 11, "c_2t": 11, "90": 11, "g_": 11, "x_iy_j": 11, "vector": 11, "frac14": 11, "leq": [11, 12, 14, 16], "v_i": [11, 12], "41": 11, "414": 11, "4142": 11, "notin": [11, 12], "10": [11, 12, 16], "complet": 11, "vert_2": [11, 13, 14], "princip": 12, "compon": 12, "analysi": 12, "pca": 12, "_d": 12, "wz": 12, "parallel": 12, "z_": 12, "i2": 12, "id": 12, "const": 12, "text": 12, "propto": 12, "limits_": [12, 13, 14, 15, 16], "xx": 12, "w_1": 12, "w_2": 12, "w_": 12, "import": [12, 16], "numpi": [12, 16], "as": [12, 16], "np": [12, 16], "matplotlib": 12, "pyplot": 12, "plt": 12, "def": [12, 16], "n_sampl": 12, "n_featur": 12, "shape": 12, "axi": 12, "scatter_matrix": 12, "transpos": 12, "eig_val": 12, "eig_vec": 12, "linalg": [12, 16], "eig": 12, "eig_pair": 12, "abs": [12, 16], "sort": 12, "revers": 12, "true": [12, 16], "ele": 12, "data": 12, "return": [12, 16], "__name__": [12, 16], "__main__": [12, 16], "x_new": 12, "print": [12, 16], "plot": 12, "ro": 12, "marker": 12, "linspac": 12, "x1": 12, "arctan": 12, "y1": 12, "proj_dir": 12, "py": 12, "append": 12, "ix": 12, "scatter": 12, "xy": 12, "zip": 12, "annot": 12, "0f": 12, "xytext": 12, "textcoord": 12, "offset": 12, "equal": 12, "pca_svd": 12, "vt": 12, "x_new_svd": 12, "f_svd": 12, "x_eig_decom": 12, "x_svd": 12, "xu": 12, "lambda_j": 12, "gram": 12, "mercer": 12, "kpca": 12, "ko": 12, "ok": 12, "oko": 12, "paramet": 12, "dimens": 12, "project": 12, "ndimens": 12, "output": 12, "has": 12, "lesser": 12, "than": 12, "input": 12, "construct": 12, "zero": 12, "k_ij": 12, "all1": 12, "k_center": 12, "eigvector": 12, "diag": [12, 14], "len": 12, "multipl": 12, "dimension": 12, "scale": 12, "mds": 12, "dist_": 12, "z_i": 12, "z_j": 12, "2z_i": 12, "tz_j": 12, "jj": 12, "2b_": 12, "mb_": 12, "mdist_": 12, "2m": 12, "center": 12, "from": 12, "sklearn": 12, "dataset": 12, "decomposit": 12, "manifold": 12, "ris": 12, "load_iri": 12, "iri": 12, "target": 12, "subplot": 12, "121": 12, "n_compon": 12, "fit": 12, "new_x_pca": 12, "transform": 12, "122": 12, "new_x_md": 12, "fit_transform": 12, "show": 12, "isometr": 12, "isomap": 12, "2d": 12, "dijkstra": 12, "floyd": 12, "dist": 12, "nearest": 12, "neighbor": 12, "otherwis": 12, "fig": 12, "figsiz": 12, "15": [12, 16], "idx": 12, "enumer": 12, "20": 12, "100": 12, "n_neighbor": 12, "set_titl": 12, "emptyset": 12, "homeomorph": 12, "isomorph": 12, "continu": 12, "_l": 12, "ik": 12, "step": 12, "min_w": 12, "varepsilon": 12, "varepsilon_i": 12, "weight": 12, "jk": 12, "ls": 12, "vert_f": 12, "min_": 12, "zmz": 12, "zz": 12, "nabla": [12, 14, 15, 16], "div": 12, "grad": 12, "v_j": 12, "e_": [12, 13], "laplacian": 12, "f_x": 12, "4f": 12, "f_j": 12, "wf": 12, "df": 12, "2w_": 12, "_iw_": 12, "_jw_": 12, "ly": 12, "stochast": 12, "q_": 12, "nl": 12, "sum_k": 12, "sum_l": 12, "s_i": 12, "101": 12, "visual": 12, "hz": 13, "arg": [14, 16], "mu_k": [14, 16], "h_1": 14, "h_q": 14, "h_i": [14, 15], "rho_k": 14, "bz": 14, "kkt": 14, "i_": 14, "tf_0": 14, "central": 14, "path": 14, "lambda_if_i": [14, 15], "tf_i": 14, "color": 14, "red": 14, "nt": [14, 16], "2f_0": 14, "nu_": 14, "r_t": 14, "f_m": [14, 15], "r_": 14, "pri": 14, "cent": 14, "succ0": 14, "dr_t": 14, "2f_i": 14, "y_": 14, "pd": 14, "lambda_k": 14, "nu_k": 14, "not": 14, "necessar": 14, "feasitbl": 14, "eta_k": 14, "surrog": 14, "dualiti": 14, "gap": 14, "epsilon_": 14, "fea": 14, "until": 14, "tag": [15, 16], "af_1": 15, "bf_2": 15, "f_a": 15, "leq0": 15, "nu_ih_i": 15, "nu_i": 15, "j_1": 15, "j_p": 15, "j_i": 15, "supremum": 15, "j_2": 15, "j_d": 15, "succeq0": [15, 16], "le0": 15, "ge0": [15, 16], "relint": 15, "opt": 16, "pi": 16, "alpha_k": 16, "alpha_i": 16, "_0": 16, "ad": 16, "alpha_0": 16, "alpha_": 16, "13": 16, "alpha_n": 16, "beta_k": 16, "14": 16, "beta_i": 16, "linear_conj_desc": 16, "negat": 16, "descent": 16, "direct": 16, "while": 16, "dtype": 16, "float": 16, "nf_": 16, "7763568394002505e": 16, "alpha_j": 16, "beta_j": 16, "dom": 16, "67": 16, "nablaf": 16, "nabla2f": 16, "10000": 16, "1e": 16, "lambda2": 16, "newton_step": 16, "01": 16, "tf": 16, "tlambda2": 16, "cx": 16, "includ": 16, "iostream": 16, "cmath": 16, "namespac": 16, "std": 16, "doubl": 16, "main": 16, "cout": 16, "cin": 16, "endl": 16, "pow": 16, "cnt": 16, "tx": 16, "break": 16}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"indic": 0, "and": 0, "tabl": 0, "maximum": 1, "mean": 1, "discrep": 1, "embed": 1, "hilbert": 2, "rkhs": 2, "mu": 4, "sigma": 4, "schatten": 5, "qr": 6, "gram": 6, "schmidt": 6, "svd": [6, 12], "jacobian": 7, "mathbf": [7, 16], "hessian": 7, "hadamard": 8, "kroneck": 8, "hermitian": 9, "lle": 12, "sne": 12, "autoencod": 13, "lagrangian": [14, 15], "alm": 14, "admm": 14, "kkt": 15, "slater": 15, "newton": 16}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 57}, "alltitles": {"\u673a\u5668\u5b66\u4e60\u57fa\u7840": [[0, "id1"]], "\u673a\u5668\u5b66\u4e60": [[0, null]], "\u6838\u65b9\u6cd5": [[0, null]], "\u6700\u4f18\u5316": [[0, null]], "\u77e9\u9635\u5206\u6790": [[0, null]], "\u6570\u5b66\u5efa\u6a21": [[0, null]], "Indices and tables": [[0, "indices-and-tables"]], "\u6700\u5927\u5747\u503c\u5dee (Maximum Mean Discrepancy)": [[1, "maximum-mean-discrepancy"]], "\u5747\u503c\u5d4c\u5165 (Mean Embedding)": [[1, "mean-embedding"]], "\u6700\u5927\u5747\u503c\u5dee\u5206": [[1, "id1"]], "\u6700\u5927\u5747\u503c\u5dee\u5206\u4f5c\u4e3a\u4e00\u79cd\u79ef\u5206\u6982\u7387\u5ea6\u91cf": [[1, "id2"]], "\u518d\u751f\u6838\u5e0c\u5c14\u4f2f\u7279\u7a7a\u95f4": [[2, "id1"]], "\u57fa\u672c\u6982\u5ff5": [[2, "id2"]], "\u6cdb\u51fd": [[2, "id3"]], "\u6cdb\u51fd\u6781\u503c\u6c42\u89e3\u2013\u53d8\u5206\u6cd5": [[2, "id4"]], "\u542b\u53c2\u53d8\u91cf\u7684\u5b9a\u79ef\u5206": [[2, "id5"]], "\u4f8b\u5b50": [[2, "id6"]], "\u7ebf\u6027\u7b97\u5b50": [[2, "id7"]], "\u518d\u751f\u6838Hilbert\u7a7a\u95f4": [[2, "hilbert"]], "\u6c42\u503c\u6cdb\u51fd": [[2, "id8"]], "\u518d\u751f\u6838": [[2, "id9"]], "RKHS\u5177\u4f53\u5316": [[2, "rkhs"]], "\u6838\u51fd\u6570\u57fa\u7840": [[3, "id1"]], "\u6838\u51fd\u6570": [[3, "id2"]], "\u793a\u4f8b": [[3, "id3"]], "\u4ec0\u4e48\u662f\u6838\u51fd\u6570": [[3, "id4"]], "\u5e38\u7528\u6838\u51fd\u6570": [[3, "id5"]], "\u7279\u5f81\u6620\u5c04\u7684\u57fa\u672c\u8fd0\u7b97": [[3, "id6"]], "\u6295\u5f71": [[3, "id7"]], "\u4e0d\u786e\u5b9a\u6a21\u578b": [[4, "id1"]], "\u62bd\u6837\u5206\u5e03": [[4, "id2"]], "\u53c2\u6570\u533a\u95f4\u4f30\u8ba1\u4e0e\u5047\u8bbe\u68c0\u9a8c": [[4, "id3"]], "\\mu\u5747\u503c\u68c0\u6d4b": [[4, "mu"]], "\u5355\u4e2a\u603b\u4f53": [[4, "id4"]], "\u4e24\u4e2a\u603b\u4f53": [[4, "id5"]], "\\sigma\u65b9\u5dee\u6bd4\u68c0\u6d4b": [[4, "sigma"]], "\u77e9\u9635\u6027\u80fd\u6307\u6807": [[5, "id1"]], "\u4e8c\u6b21\u578b": [[5, "id2"]], "\u5e38\u7528\u6027\u8d28": [[5, "id3"]], "\u884c\u5217\u5f0f": [[5, "id4"]], "\u5b9a\u4e49": [[5, "id5"]], "\u4f59\u5b50\u5f0f": [[5, "id6"]], "\u6027\u8d28\u4e00": [[5, "id7"], [5, "id18"]], "\u6027\u8d28\u4e8c": [[5, "id8"], [5, "id19"]], "\u77e9\u9635\u5185\u79ef": [[5, "id9"]], "\u77e9\u9635\u8303\u6570": [[5, "id10"], [5, "id12"]], "\u5411\u91cf\u8303\u6570": [[5, "id11"]], "\u8bf1\u5bfc\u8303\u6570": [[5, "id13"]], "\u5143\u7d20\u5f62\u5f0f\u8303\u6570": [[5, "id14"]], "Schatten\u8303\u6570": [[5, "schatten"], [5, "id16"]], "\u6838\u8303\u6570": [[5, "id15"]], "\u8ff9": [[5, "id17"]], "\u77e9\u9635\u5206\u89e3": [[6, "id1"]], "QR\u5206\u89e3": [[6, "qr"]], "QR\u7684\u4e00\u822c\u5f62\u5f0f": [[6, "id2"]], "QR\u5206\u89e3\u7684Gram-Schmidt\u6b63\u4ea4\u5316\u65b9\u6cd5": [[6, "qrgram-schmidt"]], "\u7279\u5f81\u503c\u5206\u89e3": [[6, "id3"]], "\u65b9\u9635\u7684\u7279\u5f81\u503c\u5206\u89e3": [[6, "id4"]], "\u5bf9\u79f0\u77e9\u9635\u7684\u7279\u5f81\u503c\u5206\u89e3": [[6, "id5"]], "\u5947\u5f02\u503c\u5206\u89e3": [[6, "id6"]], "SVD\u53c2\u6570\u6c42\u89e3": [[6, "svd"]], "\u77e9\u9635\u5fae\u5206": [[7, "id1"]], "Jacobian\u77e9\u9635": [[7, "jacobian"]], "\u68af\u5ea6\u77e9\u9635": [[7, "id2"]], "\u504f\u5bfc\u548c\u68af\u5ea6\u8ba1\u7b97": [[7, "id3"]], "\u57fa\u672c\u89c4\u5219": [[7, "id4"]], "\u72ec\u7acb\u6027\u5047\u8bbe": [[7, "id5"]], "\u6848\u4f8b": [[7, "id6"], [7, "id11"], [16, "id7"]], "\u4e00\u9636\u5b9e\u77e9\u9635\u5fae\u5206": [[7, "id7"]], "\u6027\u8d28": [[7, "id8"]], "\u6807\u91cf\u51fd\u6570f(\\mathbf{x})\u7684\u5411\u91cf\u53d8\u5143\\mathbf{x}\u5168\u5fae\u5206\u6c42\u504f\u5bfc\u65b9\u6cd5": [[7, "f-mathbf-x-mathbf-x"]], "\u6807\u91cf\u51fd\u6570f(\\mathbf{X})\u7684\u77e9\u9635\u91cf\u53d8\u5143\\mathbf{X}\u5168\u5fae\u5206\u6c42\u504f\u5bfc\u65b9\u6cd5": [[7, "id9"]], "\u6c42\u5bfc\u65b9\u6cd5": [[7, "id10"]], "\u5b9e\u77e9\u9635\u51fd\u6570\u7684\u504f\u5bfc\u8ba1\u7b97": [[7, "id12"]], "Hessian\u77e9\u9635": [[7, "hessian"]], "\u77e9\u9635\u8fd0\u7b97": [[8, "id1"]], "\u76f4\u548c": [[8, "id2"]], "Hadamard\u79ef": [[8, "hadamard"]], "Kronecker\u79ef": [[8, "kronecker"]], "\u5411\u91cf\u5316": [[8, "id3"]], "\u7279\u6b8a\u77e9\u9635": [[9, "id1"]], "Hermitian\u77e9\u9635": [[9, "hermitian"]], "\u9149\u77e9\u9635": [[9, "id2"]], "\u6b63\u4ea4\u77e9\u9635": [[9, "id3"]], "\u5b50\u7a7a\u95f4\u5206\u6790": [[10, "id1"]], "\u4ec0\u4e48\u662f\u5b50\u7a7a\u95f4": [[10, "id2"]], "\u6982\u5ff5": [[10, "id3"], [11, "id2"]], "\u6b63\u4ea4\u8865": [[10, "id4"]], "\u6b63\u4ea4\u6295\u5f71": [[10, "id5"]], "\u5217\uff08\u884c\uff09\u7a7a\u95f4\u4e0e\u96f6\u7a7a\u95f4": [[10, "id6"]], "\u5b50\u7a7a\u95f4\u57fa\u7684\u6784\u9020": [[10, "id7"]], "\u7a7a\u95f4\u57fa\u6784\u9020\u7684\u5947\u5f02\u503c\u5206\u89e3\u65b9\u6cd5": [[10, "id8"]], "\u5411\u91cf\u7a7a\u95f4": [[11, "id1"], [11, "id3"]], "\u5185\u79ef\u7a7a\u95f4": [[11, "id4"]], "\u8d4b\u8303\u7a7a\u95f4": [[11, "id5"]], "\u6570\u636e\u964d\u7ef4": [[12, "id1"]], "\u4e3b\u6210\u5206\u5206\u6790": [[12, "id2"]], "\u6700\u8fd1\u91cd\u6784\u6027": [[12, "id3"]], "\u6700\u5927\u53ef\u5206\u6027": [[12, "id4"]], "\u4f18\u5316\u95ee\u9898\u7684\u6c42\u89e3": [[12, "id5"]], "\u7b97\u6cd51\u2013\u7279\u5f81\u503c\u5206\u89e3": [[12, "id6"]], "\u7b97\u6cd52\u2013SVD\u5206\u89e3": [[12, "svd"]], "\u6838\u4e3b\u6210\u5206\u5206\u6790": [[12, "id7"]], "\u591a\u7ef4\u7f29\u653e": [[12, "id8"]], "\u7b97\u6cd5": [[12, "id9"], [12, "id11"]], "\u7b49\u5ea6\u91cf\u6620\u5c04": [[12, "id10"]], "\u6d41\u5f62": [[12, "id12"]], "LLE\u5c40\u90e8\u7ebf\u6027\u5d4c\u5165": [[12, "lle"]], "LLE\u57fa\u672c\u601d\u60f3": [[12, "id13"]], "LLE\u6c42\u89e3": [[12, "id14"]], "\u62c9\u666e\u62c9\u65af\u7279\u5f81\u6620\u5c04": [[12, "id15"]], "\u62c9\u666e\u62c9\u65af\u7b97\u5b50": [[12, "id16"]], "\u62c9\u666e\u62c9\u65af\u77e9\u9635": [[12, "id17"]], "\u62c9\u666e\u62c9\u65af\u53d8\u6362": [[12, "id18"]], "\u968f\u673a\u8fd1\u90bb\u5d4c\u5165": [[12, "id19"]], "SNE": [[12, "sne"]], "\u5bf9\u79f0SNE": [[12, "id20"]], "t-SNE": [[12, "t-sne"]], "\u8868\u793a\u5b66\u4e60": [[13, "id1"]], "\u81ea\u7f16\u7801\u5668(AutoEncoder)": [[13, "autoencoder"]], "\u4f18\u5316\u95ee\u9898\u6c42\u89e3(2)": [[14, "id1"]], "Lagrangian\u4e58\u5b50\u6cd5": [[14, "lagrangian"], [15, "lagrangian"]], "\u589e\u5e7fLagrangian\u4e58\u5b50\u6cd5(ALM)": [[14, "lagrangian-alm"]], "\u7b49\u5f0f\u7ea6\u675f": [[14, "id2"]], "\u6df7\u5408\u7ea6\u675f": [[14, "id3"]], "\u4ea4\u66ff\u65b9\u5411\u4e58\u5b50\u6cd5(ADMM)": [[14, "admm"]], "\u5185\u70b9\u6cd5": [[14, "id4"]], "\u969c\u788d\u6cd5": [[14, "id5"]], "\u539f\u5bf9\u5076\u5185\u70b9\u6cd5": [[14, "id6"]], "\u539f\u5bf9\u5076\u641c\u7d22\u65b9\u5411": [[14, "id7"]], "\u4ee3\u7406\u5bf9\u5076\u95f4\u9699": [[14, "id8"]], "\u51f8\u4f18\u5316\u95ee\u9898": [[15, "id1"], [15, "id5"]], "\u51f8\u51fd\u6570": [[15, "id2"]], "\u5224\u5b9a\u65b9\u6cd5": [[15, "id3"]], "\u4fdd\u51f8\u8fd0\u7b97\u4e0e\u6027\u8d28\u5224\u5b9a": [[15, "id4"]], "\u5bf9\u5076\u65b9\u6cd5": [[15, "id6"]], "KKT\u6761\u4ef6": [[15, "kkt"]], "\u5f3a\u5bf9\u5076\u6027\uff08Slater\u5b9a\u7406\uff09": [[15, "slater"]], "\u4f18\u5316\u95ee\u9898\u6c42\u89e3(1)": [[16, "id1"]], "\u4e0b\u964d\u6cd5": [[16, "id2"]], "\u6700\u901f\u4e0b\u964d\u6cd5": [[16, "id3"]], "Newton\u6cd5": [[16, "newton"]], "\u68af\u5ea6\u6295\u5f71\u6cd5": [[16, "id4"]], "\u5171\u8f6d\u68af\u5ea6\u4e0b\u964d\u6cd5": [[16, "id5"]], "\\mathbf{A}-\u5171\u8f6d": [[16, "mathbf-a"]], "\u5171\u8f6d\u68af\u5ea6\u6cd5": [[16, "id6"]], "\u4e00\u822c\u51fd\u6570\u7684\u5171\u8f6d\u68af\u5ea6\u6cd5": [[16, "id8"]], "Newton\u65b9\u6cd5": [[16, "id9"]]}, "indexentries": {}})