
<details>
<summary>Two Sum</summary>
<ul>
    Given an array of integers nums and an integer $target$, return indices of the two numbers such that they add up to $target$.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

 

Example 1:

Input: nums = $[2,7,11,15]$, target = $9$
Output: $[0,1]$
Explanation: Because $nums[0] + nums[1] == 9$, we return $[0, 1]$.
<details>
<summary>Approach</summary>
<ul>
Here, we always need to see $target - nums[i]$ if already interacted with us or not. if yes then answer exists.  
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
vector<int> twoSum(vector<int>& nums, int target) {
    map<int, int> mp;
    vector<int> ans;
    for (int i = 0; i < (int)nums.size(); ++i) {
        if (mp.count(target - nums[i])) {
            ans.push_back(i), ans.push_back(mp[target - nums[i]]); break;
        }
        mp[nums[i]] = i;
    }
    return ans;
}
```

</ul>
</details>

</ul>
</details>

 
<details>
<summary>Best Time to Buy and Sell Stock</summary>
<ul>
    You are given an array prices where $prices[i]$ is the price of a given stock on the $i$th day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return $0$.

 

Example 1:

Input: prices = $[7,1,5,3,6,4]$
Output: $5$
Explanation: Buy on day $2$ (price = $1$) and sell on day $5$ (price = $6$), profit = $6-1 = 5$.
Note that buying on day $2$ and selling on day $1$ is not allowed because you must buy before you sell.
<details>
<summary>Approach</summary>
<ul>
Here, our left pointer must be lowest possible and right pointer is highest possible. So, we initialize left to be $0$ and right to be $1$. When we find nums[left] > nums[right] we update our left pointer with right. otherwise we calculate total distance.
Final answer is the maximum total distance of left and right.
    
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
int maxProfit(vector<int>& prices) {
    int left = 0, right = 1, ma = 0;
    while (right < (int)prices.size()) {
        if (prices[left] > prices[right]) {
            left = right;
        } else {
            ma = max(ma, prices[right] - prices[left]);
        }
        right++;
    }
    return ma;
}
```

</ul>
</details>

</ul>
</details>
   
    
<details>
<summary>Find Minimum in Rotated Sorted Array</summary>
<ul>
    Suppose an array of length $n$ sorted in ascending order is rotated between $1$ and $n$ times. For example, the array nums = $[0,1,2,4,5,6,7]$ might become:

$[4,5,6,7,0,1,2]$ if it was rotated $4$ times.
$[0,1,2,4,5,6,7]$ if it was rotated $7$ times.
Notice that rotating an array $[a[0], a[1], a[2], ..., a[n-1]]$ $1$ time results in the array $[a[n-1], a[0], a[1], a[2], ..., a[n-2]]$.

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in $O(log n)$ time.

Example 1:

Input: nums = $[3,4,5,1,2]$
Output: $1$
Explanation: The original array was $[1,2,3,4,5]$ rotated $3$ times.
    
<details>
<summary>Approach</summary>
<ul>
    Here, we need to find the pivot point where there is a break. $(3, 4), (4, 5), (5, 1), (1, 2)$ here, only $(5, 1)$ point has decreasing tuple. And by observation we can see that the left portion of the pivot is always be greater and right is always smaller. So, we can run Binary Search on that Pivot point, 
```
if nums[mid] >= nums[left
    search right portion
else 
    search left portion.
```
But if the array is rotated $n$ times then it's already sorted. Here, our $nums[0]$ is the answer. So, our final result would be $min(nums[0], nums[right])$.

</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
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

</ul>
</details>

</ul>
</details>

