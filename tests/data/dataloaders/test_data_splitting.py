from contrastive_vi.data.dataloaders.data_splitting import ContrastiveDataSplitter


class TestContrastiveDataSplitter:
    def test_num_batches(
        self,
        mock_adata,
        mock_adata_background_indices,
        mock_adata_target_indices,
    ) -> None:
        train_size = 0.8
        validation_size = 0.1
        test_size = 0.1
        batch_size = 20
        n_max = max(
            len(mock_adata_background_indices),
            len(mock_adata_target_indices),
        )
        expected_train_num_batches = n_max * train_size / batch_size
        expected_val_num_batches = n_max * validation_size / batch_size
        expected_test_num_batches = n_max * test_size / batch_size

        data_splitter = ContrastiveDataSplitter(
            mock_adata,
            mock_adata_background_indices,
            mock_adata_target_indices,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )
        data_splitter.setup()
        train_dataloader = data_splitter.train_dataloader()
        val_dataloader = data_splitter.val_dataloader()
        test_dataloader = data_splitter.test_dataloader()

        assert len(train_dataloader) == expected_train_num_batches
        assert len(val_dataloader) == expected_val_num_batches
        assert len(test_dataloader) == expected_test_num_batches
