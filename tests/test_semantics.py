"""What a stage's semantic labels call a prim, as a camera's segments name it."""

from pxr import Sdf, Usd, UsdGeom, UsdSemantics

from darkroom.sensorkit import semantics


def _label(prim, *labels):
    UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr(list(labels))


def test_a_prims_labels_are_its_own_and_its_ancestors_joined():
    stage = Usd.Stage.CreateInMemory()
    _label(UsdGeom.Xform.Define(stage, "/World/Cars").GetPrim(), "vehicle")
    _label(UsdGeom.Xform.Define(stage, "/World/Cars/Wagon").GetPrim(), "car")
    UsdGeom.Mesh.Define(stage, "/World/Cars/Wagon/Body")
    UsdGeom.Mesh.Define(stage, "/World/Ground")
    label = semantics.labeller(stage, "class")
    assert label("/World/Cars/Wagon/Body") == "car, vehicle"
    assert label("/World/Cars") == "vehicle"
    assert label("/World/Ground") == "" and label("/Nowhere") == ""


def test_a_prim_in_an_instances_prototype_is_labelled_where_it_is_instanced():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/Tree/Branch")
    trees = UsdGeom.Xform.Define(stage, "/World/Trees").GetPrim()
    _label(trees, "tree")
    one = stage.DefinePrim("/World/Trees/One")
    one.GetReferences().AddInternalReference(Sdf.Path("/Tree"))
    one.SetInstanceable(True)
    branch = one.GetPrototype().GetChild("Branch")
    assert semantics.labeller(stage, "class")(str(branch.GetPath())) == "tree"
