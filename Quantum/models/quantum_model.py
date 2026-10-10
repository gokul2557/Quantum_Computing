    @classmethod
    def load(cls, model_dir: str = "models"):
        """Loads trained QuantumFraudVQC from disk."""
        dill_path = os.path.join(model_dir, "quantum_vqc.dill")
        meta_path = os.path.join(model_dir, "quantum_model_meta.json")
        weights_path = os.path.join(model_dir, "vqc_weights.npy")
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Quantum model metadata missing in {model_dir}")
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        model_obj = cls(
            num_qubits=meta.get("num_qubits", 4),
            reps=meta.get("reps", 1),
            entanglement=meta.get("entanglement", "linear"),
            optimizer_name=meta.get("optimizer_name", "COBYLA"),
            max_iter=meta.get("max_iter", 50),
            shots=meta.get("shots", 1024)
        )
        
        loaded = False
        # 1. Attempt loading serialized VQC object via dill if available
        if os.path.exists(dill_path):
            try:
                model_obj.vqc = VQC.from_dill(dill_path)
                model_obj.weights = np.array(model_obj.vqc.weights)
                model_obj.is_trained = True
                loaded = True
                print(f"[Quantum VQC] Successfully loaded VQC model from '{dill_path}'.")
            except Exception as e:
                print(f"[Quantum VQC] dill deserialization failed ({e}), falling back to direct weights...")

        # 2. Robust cross-platform fallback via saved weights array (bulletproof across Linux/Windows/Python versions)
        if not loaded and os.path.exists(weights_path):
            from types import SimpleNamespace
            weights = np.load(weights_path)
            model_obj.weights = weights
            model_obj.vqc._fit_result = SimpleNamespace(x=weights, fun=0.0)
            model_obj.is_trained = True
            loaded = True
            print(f"[Quantum VQC] Successfully restored VQC from weights array '{weights_path}'.")

        if not loaded:
            raise FileNotFoundError(f"Could not load VQC from dill or weights in {model_dir}")
            
        return model_obj
