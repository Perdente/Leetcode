
153. Find Minimum in Rotated Sorted Array
Suppose an array of length $n$ sorted in ascending order is rotated between $1$ and $n$ times. For example, the array nums = $[0,1,2,4,5,6,7]$ might become:

$[4,5,6,7,0,1,2]$ if it was rotated $4$ times.
$[0,1,2,4,5,6,7]$ if it was rotated $7$ times.
Notice that rotating an array $[a[0], a[1], a[2], ..., a[n-1]]$ $1$ time results in the array $[a[n-1], a[0], a[1], a[2], ..., a[n-2]]$.

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in $O(log n)$ time.

Example 1:

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.

Approach:
Here, we need to find the pivot point where there is a break. $(3, 4), (4, 5), (5, 1), (1, 2)$ here, only $(5, 1)$ point has decreasing tuple. And by observation we can see that the left portion of the pivot is always be greater and right is always smaller. So, we can run Binary Search on that Pivot point, 
```
if nums[mid] >= nums[left
    search right portion
else 
    search left portion.
```
But if the array is rotated $n$ times then it's already sorted. Here, our nums[0] is the answer. So, our final result would be min(nums[0], nums[right]).

```c++
int findMin(vector<int>& nums) {
    int n = nums.size();
    int left = 0, right = n - 1;
    while (right > left + 1) {
        int mid = (left + right) / 2;
        if (nums[mid] >= nums[left]) {
            left = mid;
        } else {
            right = mid;
        }
    }

    return min(nums[0], nums[right]);
}
```