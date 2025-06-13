print("--- END DEBUGGING ---\n")

    # --- Final Saving Stage ---
    
    # 1. Define a temporary output directory on the local node disk.
    #    This path will be visible because we bind /tmp in the shell script.
    local_output_dir = f"/tmp/{os.environ['SLURM_JOB_ID']}/final_output"
    
    # 2. Create the directory
    os.makedirs(local_output_dir, exist_ok=True)
    print(f"\n--- Saving temporary artifacts to local disk: {local_output_dir} ---")

    # 3. Save all artifacts to this LOCAL directory
    lightweight_path = os.path.join(local_output_dir, "lightweight_network.pt")
    torch.save(lightweight_network.state_dict(), lightweight_path)

    pruned_model.save_pretrained(local_output_dir)
    tokenizer.save_pretrained(local_output_dir)
    pruned_model.config.save_pretrained(local_output_dir)

    print(f"✅ All artifacts successfully saved locally.")