# Cell 2 (revisi): Implementasi Kelas StockGMMHMM dengan perbaikan
class StockGMMHMM:
    """
    Kelas untuk implementasi model GMM-HMM untuk prediksi pasar saham.
    """
    def __init__(self, n_states=3, n_mix=3, covariance_type='diag', random_state=42):
        """
        Inisialisasi model GMM-HMM.
        
        Parameters:
        -----------
        n_states: int
            Jumlah state tersembunyi dalam model HMM
        n_mix: int
            Jumlah komponen Gaussian mixture per state
        covariance_type: str
            Tipe kovarians ('full', 'diag', 'tied', atau 'spherical')
        random_state: int
            Seed untuk reproducibility
        """
        self.n_states = n_states
        self.n_mix = n_mix
        self.covariance_type = covariance_type  # Default ke 'diag' untuk stabilitas
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df, selected_features=None):
        """
        Preprocessing data untuk model GMM-HMM.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame yang berisi data saham
        selected_features: list
            List fitur yang akan digunakan. Jika None, gunakan semua fitur yang tersedia
            
        Returns:
        --------
        X_scaled: numpy.ndarray
            Data yang telah dipreprocessing
        """
        # Default features jika tidak ada yang ditentukan
        if selected_features is None:
            selected_features = [
                'open', 'high', 'low', 'close', 'Volume',
                'Histogram', 'MACD', 'Signal', 'K', 'D',
                'Turnover (Cr)', 'Turnover / 10MA (X)'
            ]
        
        # Ekstrak fitur dan tangani missing values
        X = df[selected_features].copy()
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Scale fitur
        X_scaled = self.scaler.fit_transform(X)
        
        # Tambahkan sedikit noise untuk menghindari singularitas
        np.random.seed(self.random_state)
        X_scaled += np.random.normal(0, 1e-6, X_scaled.shape)
        
        return X_scaled
    
    def train(self, X):
        """
        Melatih model GMM-HMM.
        
        Parameters:
        -----------
        X: numpy.ndarray
            Data terpreprocessing untuk pelatihan
            
        Returns:
        --------
        self: StockGMMHMM
            Instance model yang telah dilatih
        """
        # Inisialisasi model GMM-HMM dengan parameter yang lebih stabil
        self.model = hmm.GMMHMM(
            n_components=self.n_states,
            n_mix=self.n_mix,
            covariance_type=self.covariance_type,  # Gunakan 'diag' untuk stabilitas
            random_state=self.random_state,
            n_iter=100,
            tol=0.01,
            init_params='kmeans',  # Gunakan kmeans untuk inisialisasi
            params='stmcw',  # Perbarui semua parameter
            verbose=True
        )
        
        try:
            # Coba latih model
            self.model.fit(X)
        except Exception as e:
            print(f"Error dalam pelatihan: {str(e)}")
            print("Mencoba dengan covariance_type='diag' dan covars_prior=0.1")
            
            # Jika gagal, coba dengan parameter yang lebih stabil
            self.covariance_type = 'diag'
            self.model = hmm.GMMHMM(
                n_components=self.n_states,
                n_mix=self.n_mix,
                covariance_type='diag',
                random_state=self.random_state,
                n_iter=100,
                tol=0.1,  # Tingkatkan toleransi
                init_params='kmeans',
                params='stmcw',
                verbose=True
            )
            
            # Tambahkan regularisasi ke matriks kovarians
            self.model.covars_prior = 0.1
            self.model.fit(X)
        
        return self
    
    def predict_states(self, X):
        """
        Memprediksi state tersembunyi untuk data input.
        
        Parameters:
        -----------
        X: numpy.ndarray
            Data terpreprocessing untuk prediksi
            
        Returns:
        --------
        states: numpy.ndarray
            State tersembunyi yang diprediksi
        state_probs: numpy.ndarray
            Probabilitas state
        """
        try:
            # Prediksi state
            states = self.model.predict(X)
            
            # Hitung probabilitas state
            logprob, state_probs = self.model.score_samples(X)
            
            return states, state_probs
        except Exception as e:
            print(f"Error dalam prediksi: {str(e)}")
            print("Mencoba dengan pendekatan alternatif...")
            
            # Alternatif: Hitung probabilitas secara manual
            log_probs = np.zeros((len(X), self.n_states))
            
            # Hitung log probabilitas untuk setiap state
            for i in range(self.n_states):
                # Untuk setiap mixture, hitung probabilitas
                for j in range(self.n_mix):
                    # Hitung mahalanobis distance
                    means = self.model.means_[i, j]
                    
                    if self.covariance_type == 'diag':
                        covars = self.model.covars_[i, j]
                        precision = 1.0 / covars
                        log_det = np.sum(np.log(covars))
                    else:
                        # Untuk tipe kovarians lain, gunakan implementasi sederhana
                        precision = np.eye(X.shape[1])
                        log_det = 0
                    
                    # Hitung log probabilitas untuk mixture ini
                    for k in range(len(X)):
                        diff = X[k] - means
                        log_probs[k, i] += self.model.weights_[i, j] * np.exp(
                            -0.5 * np.sum(diff**2 * precision) - 0.5 * log_det
                        )
            
            # Normalisasi dan temukan state dengan probabilitas tertinggi
            log_probs = log_probs / np.sum(log_probs, axis=1, keepdims=True)
            states = np.argmax(log_probs, axis=1)
            
            return states, log_probs