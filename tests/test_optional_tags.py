"""Exercise node methods without requiring a running ComfyUI or media codecs."""

import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock


def load_node(filename):
    source = Path(__file__).resolve().parents[1] / "nodes" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    # Keep the actual class implementation, replacing only external dependencies.
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    ast.fix_missing_locations(tree)
    namespace = {
        "EagleFeederBase": object,
        "build_annotation": MagicMock(return_value="test annotation"),
        "EagleAPI": MagicMock(),
        "tensor_to_pil": MagicMock(),
        "IO": MagicMock(),
        "VideoContainer": MagicMock(),
        "VideoCodec": MagicMock(),
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace[tree.body[1].name], namespace


class TestOptionalTags(unittest.TestCase):
    def test_nodes_accept_missing_and_connected_tags(self):
        for filename in ["eagle_feeder_png.py", "eagle_feeder_mp4.py", "eagle_feeder_animated_webp.py"]:
            for tags in [None, "", "cat,, dog"]:
                with self.subTest(filename=filename, tags=tags):
                    cls, namespace = load_node(filename)
                    inputs = cls.INPUT_TYPES()
                    self.assertNotIn("tags", inputs["required"])
                    self.assertTrue(inputs["optional"]["tags"][1]["forceInput"])
                    node = cls()
                    node.img_dir = "unused"
                    node.find_id_by_name = MagicMock(return_value="folder")
                    node.get_file_name = MagicMock(return_value="test")
                    kwargs = dict(folder_name="", eagle_host="host", eagle_token="", file_server_host="localhost", embed_workflow=False)
                    if filename == "eagle_feeder_png.py":
                        kwargs = {key: [value] for key, value in kwargs.items()}
                        kwargs.update(images=[[MagicMock(), MagicMock()]], prompt=[None], extra_pnginfo=[None])
                        if tags is not None:
                            kwargs["tags"] = [tags]
                    else:
                        if tags is not None:
                            kwargs["tags"] = tags
                        if filename == "eagle_feeder_mp4.py":
                            kwargs.update(video=MagicMock(), format="auto", codec="auto")
                        else:
                            kwargs.update(images=[MagicMock()], fps=6, lossless=True, quality=80, method="default")
                    node.send_to_eagle(**kwargs)
                    calls = namespace["EagleAPI"].return_value.add_from_url.call_args_list
                    self.assertEqual(len(calls), 2 if filename == "eagle_feeder_png.py" else 1)
                    expected = tags.split(",") if tags else []
                    for call in calls:
                        self.assertEqual(call.kwargs["annotation"], "test annotation")
                        # PNG's connected empty string is filtered at the API boundary.
                        self.assertEqual([t for t in call.args[1] if t], [t for t in expected if t])


if __name__ == "__main__":
    unittest.main()
