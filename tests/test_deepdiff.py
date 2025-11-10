from deepdiff import DeepDiff
import json

# Example 1: Simple value changes
print("=" * 60)
print("Example 1: Value Changes")
print("=" * 60)
json1 = {"name": "John", "age": 30, "city": "NYC"}
json2 = {"name": "John", "age": 31, "city": "NYC"}
diff = DeepDiff(json1, json2)
print(f"JSON1: {json1}")
print(f"JSON2: {json2}")
print(f"Diff: {diff}\n")

# Example 2: Structure change (nested vs flat)
print("=" * 60)
print("Example 2: Structure Change (Nested vs Flat)")
print("=" * 60)
json1 = {"user": {"name": "John", "age": 30}}
json2 = {"user_name": "John", "user_age": 30}
diff = DeepDiff(json1, json2)
print(f"JSON1: {json1}")
print(f"JSON2: {json2}")
print(f"Diff: {diff}\n")

# Example 3: Field rename (same structure)
print("=" * 60)
print("Example 3: Field Rename")
print("=" * 60)
json1 = {"name": "John", "age": 30}
json2 = {"full_name": "John", "age": 30}
diff = DeepDiff(json1, json2)
print(f"JSON1: {json1}")
print(f"JSON2: {json2}")
print(f"Diff: {diff}\n")

# Example 4: Tree view
print("=" * 60)
print("Example 4: Tree View")
print("=" * 60)
json1 = {"user": {"name": "John", "age": 30}}
json2 = {"user": {"name": "Jane", "age": 30}}
diff = DeepDiff(json1, json2, view='tree')
print(f"JSON1: {json1}")
print(f"JSON2: {json2}")
print(f"Diff type: {type(diff)}")
print(f"Diff: {diff}")
if 'values_changed' in diff:
    for change in diff['values_changed']:
        print(f"  Path: {change.path()}")
        print(f"  Old: {change.t1}")
        print(f"  New: {change.t2}")
print()

# Example 5: Array reordering
print("=" * 60)
print("Example 5: Array Order (ignore_order=False)")
print("=" * 60)
json1 = {"items": [1, 2, 3]}
json2 = {"items": [3, 2, 1]}
diff = DeepDiff(json1, json2)
print(f"JSON1: {json1}")
print(f"JSON2: {json2}")
print(f"Diff: {diff}\n")

print("=" * 60)
print("Example 6: Array Order (ignore_order=True)")
print("=" * 60)
diff = DeepDiff(json1, json2, ignore_order=True)
print(f"JSON1: {json1}")
print(f"JSON2: {json2}")
print(f"Diff: {diff}\n")
