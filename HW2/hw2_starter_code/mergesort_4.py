def merge_4(A, B, C, D):
    """Merge 4 arrays passed in sorted order."""
    # TODO: Your code here.
    def merge_2(arr1, arr2):
        if len(arr1)==0:
            return arr2
        if len(arr2)==0:
            return arr1
        i, j = 0, 0
        sorted_arr = []
        while i<len(arr1) and j<len(arr2):
            if arr1[i]<arr2[j]:
                sorted_arr.append(arr1[i])
                i+=1
            else:
                sorted_arr.append(arr2[j])
                j+=1
        if i<len(arr1):
            sorted_arr.extend(arr1[i:])
        if j<len(arr2):
            sorted_arr.extend(arr2[j:])
        return sorted_arr[:]
    
    sorted_arr1 = merge_2(A, B)
    sorted_arr2 = merge_2(C, D)
    return merge_2(sorted_arr1, sorted_arr2)



def mergesort_4(arr):
    """Sort `arr` using mergesort with 4 instead of 2 subproblems."""
    # TODO: Your code here. You should use `merge_4` to combine
    # solutions from subproblems.
    if len(arr)<=1:
        return arr
    length = len(arr)
    q1 = length//4
    q2 = length//2
    q3 = 3*length//4
    A = mergesort_4(arr[:q1])
    B = mergesort_4(arr[q1:q2])
    C = mergesort_4(arr[q2:q3])
    D = mergesort_4(arr[q3:])
    return merge_4(A, B, C, D)


A = [2, 1, 5, 6, 2, 4, 6]
res = mergesort_4(A)
print(res)