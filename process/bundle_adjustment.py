def run_global_bundle_adjustment(sparse_dir):

    print("\n" + "=" * 50)
    print("RUNNING GLOBAL BUNDLE ADJUSTMENT")
    print("=" * 50)

    if not (sparse_dir / "cameras.bin").exists():
        print("Error: Model files not found for BA.")
        return

    reconstruction = pycolmap.Reconstruction(sparse_dir)

    options = pycolmap.BundleAdjustmentOptions()
    options.solver_options.num_threads = 8
    options.solver_options.max_num_iterations = 100

    print(f"Initial Mean Reprojection Error: "
          f"{reconstruction.compute_mean_reprojection_error():.4f} pixels")

    pycolmap.bundle_adjustment(reconstruction, options)

    print(f"Final Mean Reprojection Error: "
          f"{reconstruction.compute_mean_reprojection_error():.4f} pixels")

    reconstruction.write(sparse_dir)
    print("✓ Model successfully refined and saved.")
    print("=" * 50 + "\n")
