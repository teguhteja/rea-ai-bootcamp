from typing import List
from itertools import combinations

def twoSum(nums: List[int], target: int) -> List[int]:
    retval = ''
    combs = list(combinations(list(range(len(nums))), 2))
    for c in combs:
        if nums[c[0]] + nums[c[1]] == target:
            retval = [c[0],c[1]]
            break
    return retval 

print(twoSum(nums = [3,4,5,6], target = 7)) #  [0,1]
print(twoSum(nums = [4,5,6], target = 10)) #  [0,2]
print(twoSum(nums = [5,5], target = 10)) #  [0,2]
print(twoSum(nums=[1,3,4,2],target=6)) # [2,3]