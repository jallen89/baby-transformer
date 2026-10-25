"""Small, repeatable benchmark harness for transformer experiments."""

import math
import os
import re
import time

import pandas as pd
import torch
from torch import nn, optim


class BenchmarkHarness:
    START_MARKER = "<!-- BENCHMARK_START -->"
    END_MARKER = "<!-- BENCHMARK_END -->"

    def __init__(
        self,
        tokenizer,
        train_loader,
        val_loader,
        total_steps=500,
        lr=1e-3,
        seed=42,
    ):
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_steps = total_steps
        self.lr = lr
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.results = []

    def _reset_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _evaluate(self, model, max_batches=50):
        model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for idx, (batch_x, batch_y) in enumerate(self.val_loader):
                if max_batches and idx >= max_batches:
                    break
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = model(batch_x)
                batch_size, seq_len, vocab_size = logits.shape
                loss = self.criterion(
                    logits.view(batch_size * seq_len, vocab_size),
                    batch_y.view(batch_size * seq_len),
                )
                total_loss += loss.item()
                total_batches += 1
        return total_loss / max(1, total_batches)

    def run_candidate(
        self,
        name: str,
        model_builder_fn,
        metadata: dict = None,
        context="Thy king is here",
    ):
        self._reset_seed()
        model = model_builder_fn().to(self.device)
        total_params = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=0.01
        )
        data_iter = iter(self.train_loader)

        best_val_loss = float("inf")
        train_losses = []
        start_time = time.time()

        print(f"--> Benchmarking: {name} ({total_params:,} params)")

        for step in range(1, self.total_steps + 1):
            try:
                batch_x, batch_y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch_x, batch_y = next(data_iter)

            model.train()
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            logits = model(batch_x)
            batch_size, seq_len, vocab_size = logits.shape
            loss = self.criterion(
                logits.view(batch_size * seq_len, vocab_size),
                batch_y.view(batch_size * seq_len),
            )
            train_losses.append(loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 50 == 0 or step == self.total_steps:
                val_loss = self._evaluate(model)
                best_val_loss = min(best_val_loss, val_loss)

        wall_time = time.time() - start_time
        final_val_loss = self._evaluate(model)
        final_train_loss = sum(train_losses) / max(1, len(train_losses))
        val_ppl = (
            math.exp(final_val_loss)
            if final_val_loss < 20
            else float("inf")
        )

        run_entry = {
            "Model": name,
            "Positional Emb": (
                metadata.get("pos_emb", "None") if metadata else "None"
            ),
            "Attention Type": (
                metadata.get("attn_type", "Single-Head")
                if metadata
                else "Single-Head"
            ),
            "Norm": metadata.get("norm", "None") if metadata else "None",
            "Parameters": f"{total_params:,}",
            "Train Loss": round(final_train_loss, 4),
            "Best Val Loss": round(best_val_loss, 4),
            "Final Val Loss": round(final_val_loss, 4),
            "Final PPL": round(val_ppl, 1),
            "Steps/sec": round(self.total_steps / wall_time, 2),
            "Completion": self.generate_text(model, context),
        }
        self.results.append(run_entry)

    @classmethod
    def _read_results_from_readme(cls, content):
        pattern = (
            rf"{re.escape(cls.START_MARKER)}(.*?){re.escape(cls.END_MARKER)}"
        )
        match = re.search(pattern, content, flags=re.DOTALL)
        if not match:
            return []

        table_lines = [
            line.strip()
            for line in match.group(1).splitlines()
            if line.strip().startswith("|")
        ]
        if len(table_lines) < 3:
            return []

        headers = [value.strip() for value in table_lines[0][1:-1].split("|")]
        results = []
        for line in table_lines[2:]:
            # Completion is the final column, so maxsplit preserves any pipes
            # that may appear in generated text there.
            values = [
                value.strip()
                for value in line[1:-1].split("|", len(headers) - 1)
            ]
            if len(values) != len(headers):
                continue

            result = dict(zip(headers, values))
            if result.get("Model"):
                results.append(result)
            elif results and result.get("Completion"):
                # Recover continuation lines produced by older versions when
                # generated text contained newline characters.
                results[-1]["Completion"] = (
                    results[-1].get("Completion", "")
                    + " "
                    + result["Completion"]
                ).strip()
        return results

    @staticmethod
    def _clean_result(result):
        cleaned = {}
        for key, value in result.items():
            if isinstance(value, str):
                # Markdown tables require every record to occupy one line.
                value = re.sub(r"\s+", " ", value).strip()
                # A literal pipe would create an additional Markdown column.
                value = value.replace("|", "¦")
            cleaned[key] = value
        return cleaned

    def summary(self, readme_path="../README.md"):
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as readme_file:
                content = readme_file.read()
        else:
            content = ""

        merged_results = [
            self._clean_result(result)
            for result in self._read_results_from_readme(content)
        ]
        model_indexes = {
            str(result.get("Model")): index
            for index, result in enumerate(merged_results)
        }

        for raw_result in self.results:
            result = self._clean_result(raw_result)
            model_name = str(result.get("Model"))
            if model_name in model_indexes:
                merged_results[model_indexes[model_name]] = result
            else:
                model_indexes[model_name] = len(merged_results)
                merged_results.append(result)

        df = pd.DataFrame(merged_results)
        md_table = df.to_markdown(index=False)

        print("\n" + "=" * 35 + " BENCHMARK LEADERBOARD " + "=" * 35)
        print(md_table)
        print("=" * 93 + "\n")

        start_marker = self.START_MARKER
        end_marker = self.END_MARKER
        section_content = (
            f"{start_marker}\n## Model Benchmark Results\n\n"
            f"{md_table}\n{end_marker}"
        )

        if content:
            if start_marker in content and end_marker in content:
                pattern = rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}"
                updated_content = re.sub(
                    pattern, section_content, content, flags=re.DOTALL
                )
            else:
                updated_content = content.rstrip() + f"\n\n{section_content}\n"
        else:
            updated_content = f"# Transformer Experiments\n\n{section_content}\n"

        with open(readme_path, "w", encoding="utf-8") as readme_file:
            readme_file.write(updated_content)

        print(f"Benchmark results successfully written to {readme_path}")
        return df

    def generate_text(self, model, prompt=None, max_new_tokens=40):
        prompt = prompt or getattr(self, "context", "Thy king is here")
        model.eval()
        with torch.no_grad():
            encoded = self.tokenizer.encode(prompt)
            if isinstance(encoded, dict):
                encoded = encoded.get("input_ids", encoded)

            if isinstance(encoded, torch.Tensor):
                input_ids = encoded.to(self.device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
            else:
                input_ids = torch.tensor(
                    [encoded], dtype=torch.long, device=self.device
                )

            try:
                out_ids = model.generate(
                    input_ids, max_seq_len=max_new_tokens
                )
            except TypeError:
                out_ids = model.generate(
                    input_ids,
                    max_seq_len=input_ids.shape[1] + max_new_tokens,
                )

            out_ids = out_ids[0].detach().cpu().tolist()
            completion = self.tokenizer.decode(out_ids)
        return prompt + " " + completion
