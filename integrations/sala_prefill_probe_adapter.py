"""Worker-side native-prefill graph evidence. Install before Engine creation."""
import dataclasses
import json
import os
from pathlib import Path


def install(control_path, events_path):
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import BreakableCudaGraphBackend
    if getattr(PrefillCudaGraphRunner, '_sala_probe', False): return
    if os.environ.get('SALA_PREFILL_SHAPE_FIX') == '1':
        # Output norm is on MiniCPMLightningMixer, not MiniCPMAttention.
        from sglang.srt.models.minicpm import MiniCPMLightningMixer
        init_original = MiniCPMLightningMixer.__init__
        def mixer_init(self, *args, **kwargs):
            init_original(self, *args, **kwargs)
            if not self.use_output_norm:
                return
            width = self.o_norm.weight.numel()
            if width != self.num_heads * self.head_dim:
                raise RuntimeError('Unexpected MiniCPM output norm width')
            forward_original = self.attn.forward
            def attention_forward(*a, **kw):
                result = forward_original(*a, **kw)
                if result.ndim == 2 and result.shape[-1] == width:
                    return result
                if (result.ndim == 3 and result.shape[1] == self.num_heads
                        and result.shape[2] == self.head_dim
                        and result.shape[1] * result.shape[2] == width):
                    return result.reshape(result.shape[0], width)
                raise RuntimeError('Unexpected MiniCPM attention output shape')
            self.attn.forward = attention_forward
        MiniCPMLightningMixer.__init__ = mixer_init
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    forward_original = ModelRunner.forward
    pending = []
    def forward(self, forward_batch, *args, **kwargs):
        c = json.loads(Path(control_path).read_text())
        if pending and pending[0][0]['run_id'] != c['run_id']:
            with open(events_path, 'a', encoding='utf-8') as stream:
                for event, start, end in pending:
                    end.synchronize()
                    event['cuda_ms'] = start.elapsed_time(end)
                    stream.write(json.dumps(event)+'\n')
            pending.clear()
        eligible = (c.get('profile', False) and c.get('run_id') != 'initialization'
            and forward_batch.forward_mode == ForwardMode.EXTEND
            and forward_batch.input_ids.numel() == c.get('tokens'))
        if not eligible:
            return forward_original(self, forward_batch, *args, **kwargs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = forward_original(self, forward_batch, *args, **kwargs)
        end.record()
        pending.append((dict(run_id=c['run_id'], kind='prefill_forward_cuda_interval',
            tokens=forward_batch.input_ids.numel(), method='ModelRunner.forward',
            scope='current-stream model forward GPU timeline including launch gaps; not kernel sum'),start,end))
        return result
    ModelRunner.forward = forward
    execute_original = PrefillCudaGraphRunner.execute
    replay_original = BreakableCudaGraphBackend.replay
    active = []
    def replay(self, shape_key, static_forward_batch, **kwargs):
        result = replay_original(self, shape_key, static_forward_batch, **kwargs)
        # The installed backend directly calls _graphs[shape_key].replay();
        # only its successful return inside the runner's execute counts.
        if active and active[-1].backend is self:
            from sglang.srt.runtime_context import get_exec
            c = json.loads(Path(control_path).read_text())
            event = dict(run_id=c['run_id'],kind='actual_prefill_graph_replay',
                runner_class=type(active[-1]).__name__, method='execute -> backend.replay',
                backend=active[-1].prefill_backend_name,
                shape_adapter=('restore eager 2D attention contract' if os.environ.get('SALA_PREFILL_SHAPE_FIX') == '1' else 'disabled'),
                resolved_graph_config=dataclasses.asdict(get_exec().graph.cuda_graph_config))
            with open(events_path,'a',encoding='utf-8') as stream:
                stream.write(json.dumps(event)+'\n')
        return result
    def execute(self, forward_batch, **kwargs):
        active.append(self)
        try:return execute_original(self, forward_batch, **kwargs)
        finally:active.pop()
    BreakableCudaGraphBackend.replay = replay
    PrefillCudaGraphRunner.execute = execute
    PrefillCudaGraphRunner._sala_probe = True
