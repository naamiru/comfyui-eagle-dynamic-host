import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('annotation', Path(__file__).resolve().parents[1] / 'utils/annotation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
build = module.build_annotation


def node(kind, **inputs):
    return {'class_type': kind, 'inputs': inputs}


def graph():
    return {
        '1': node('CheckpointLoaderSimple', ckpt_name='model.safetensors'),
        '2': node('CLIPTextEncode', text='a cat'),
        '3': node('CLIPTextEncode', text='blurry'),
        '4': node('KSampler', model=['1', 0], positive=['2', 0], negative=['3', 0], steps=20, sampler_name='euler', scheduler='normal', cfg=7, seed=0),
        '5': node('VAEDecode', samples=['4', 0]),
        '6': node('EagleFeederPng', images=['5', 0]),
    }


class AnnotationTests(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(build(graph(), '6', size=(512, 768)), 'a cat\n\nNegative prompt:blurry\nSteps: 20, Sampler: euler normal, CFG scale: 7, Seed: 0, Size: 512x768, Model: model.safetensors')

    def test_override_and_empty_negative(self):
        result = build(graph(), '6', positive='a dog', negative='')
        self.assertTrue(result.startswith('a dog\nSteps:'))
        self.assertNotIn('blurry', result)

    def test_unknown_values_omitted(self):
        g = graph()
        g['9'] = node('Compute', value=123)
        g['4']['inputs']['seed'] = ['9', 0]
        g['2']['inputs']['text'] = ['9', 0]
        result = build(g, '6')
        self.assertNotIn('Seed:', result)
        self.assertNotIn('123', result)
        self.assertNotIn('None', result)

    def test_unrelated_sampler_ignored(self):
        g = graph(); g['9'] = node('KSampler', seed=999)
        self.assertNotIn('999', build(g, '6'))
        self.assertEqual(build(g, 'missing'), '')

    def test_ambiguous_branches_and_cycles(self):
        g = graph()
        g['7'] = node('KSampler', seed=999)
        g['8'] = node('ImageBatch', image1=['4', 0], image2=['7', 0], cycle=['8', 0])
        g['6']['inputs']['images'] = ['8', 0]
        self.assertEqual(build(g, '6'), '')

    def test_advanced_sampler(self):
        g = graph(); g['4']['class_type'] = 'KSamplerAdvanced'
        del g['4']['inputs']['seed']; g['4']['inputs']['noise_seed'] = 42
        self.assertIn('Seed: 42', build(g, '6'))

    def test_missing_graph(self):
        self.assertEqual(build(None, positive='cat'), 'cat')
        self.assertEqual(build(None), '')

    def test_d2_runtime_pipe_not_guessed(self):
        g = graph(); g['4']['class_type'] = 'D2 KSampler'
        g['4']['inputs']['d2_pipe'] = ['1', 0]
        self.assertEqual(build(g, '6'), '')

    def test_quoted_model(self):
        g = graph(); g['1']['inputs']['ckpt_name'] = 'model, v2.safetensors'
        self.assertIn('Model: "model, v2.safetensors"', build(g, '6'))
