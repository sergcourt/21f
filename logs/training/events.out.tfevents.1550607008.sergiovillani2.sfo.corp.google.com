       �K"	   ��Abrain.Event:2$�O�M�      ��	� ��A"��
t
input/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7��*
dtype0*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
layer_1/weights1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
layer_1/weights1/readIdentitylayer_1/weights1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
!layer_1/biases1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
layer_1/biases1
VariableV2*"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_1/biases1/readIdentitylayer_1/biases1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*
T0*'
_output_shapes
:���������
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:���������
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_2/weights2*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_2/weights2/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_2/weights2*
valueB
 *׳]�
�
/layer_2/weights2/Initializer/random_uniform/maxConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_2/weights2*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
layer_2/weights2
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container 
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
layer_2/weights2/readIdentitylayer_2/weights2*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
!layer_2/biases2/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
layer_2/biases2
VariableV2*"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*
T0*'
_output_shapes
:���������
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:���������
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@layer_3/weights3*
valueB"      
�
/layer_3/weights3/Initializer/random_uniform/minConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/maxConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_3/weights3*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_3/weights3
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
layer_3/weights3
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_3/weights3/readIdentitylayer_3/weights3*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
layer_3/biases3
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
z
layer_3/biases3/readIdentitylayer_3/biases3*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*
T0*'
_output_shapes
:���������
S
layer_3/ReluRelulayer_3/add*
T0*'
_output_shapes
:���������
�
0output/weights4/Initializer/random_uniform/shapeConst*"
_class
loc:@output/weights4*
valueB"      *
dtype0*
_output_shapes
:
�
.output/weights4/Initializer/random_uniform/minConst*"
_class
loc:@output/weights4*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/maxConst*"
_class
loc:@output/weights4*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*"
_class
loc:@output/weights4*
seed2 
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
output/weights4
VariableV2*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
~
output/weights4/readIdentityoutput/weights4*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
output/biases4
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
w
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
g

output/addAddoutput/MatMuloutput/biases4/read*
T0*'
_output_shapes
:���������
s
cost/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*
T0*'
_output_shapes
:���������
[

cost/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
^
train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/cost/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/cost/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
&train/gradients/cost/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
_output_shapes
: *
T0
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*
T0*'
_output_shapes
:���������
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
_output_shapes
:*
T0*
out_type0
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
_output_shapes
:*
T0*
out_type0
q
'train/gradients/output/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul*'
_output_shapes
:���������
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_3/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*'
_output_shapes
:���������*
T0
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
train/beta1_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
z
train/beta1_power/readIdentitytrain/beta1_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *"
_class
loc:@layer_1/biases1*
valueB
 *w�?
�
train/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
�
train/layer_1/weights1/Adam_1
VariableV2*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_1/biases1*
valueB*    
�
train/layer_1/biases1/Adam
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam_1
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam
VariableV2*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam_1
VariableV2*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam
VariableV2*"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam_1
VariableV2*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container 
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
,train/output/weights4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*"
_class
loc:@output/weights4*
valueB*    
�
train/output/weights4/Adam
VariableV2*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam_1
VariableV2*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
+train/output/biases4/Adam/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam
VariableV2*!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
-train/output/biases4/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam_1
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_2/weights2
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_2/biases2*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_3/biases3*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*"
_class
loc:@output/weights4
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@output/biases4*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking( 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking( 
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*%
valueB Blogging/current_cost*
dtype0*
_output_shapes
: 
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
T0*
_output_shapes
: 
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
�
save/SaveV2/tensor_namesConst*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign"��߳�     ��d]	v���AJ܉
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.12.02v1.12.0-0-ga6d8ffae09��
t
input/PlaceholderPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7��*
dtype0*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
layer_1/weights1/readIdentitylayer_1/weights1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
!layer_1/biases1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
layer_1/biases1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
z
layer_1/biases1/readIdentitylayer_1/biases1*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*
T0*'
_output_shapes
:���������
S
layer_1/ReluRelulayer_1/add*'
_output_shapes
:���������*
T0
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_2/weights2*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_2/weights2/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_2/weights2*
valueB
 *׳]�
�
/layer_2/weights2/Initializer/random_uniform/maxConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
layer_2/weights2
VariableV2*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
layer_2/weights2/readIdentitylayer_2/weights2*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
!layer_2/biases2/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
layer_2/biases2
VariableV2*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*'
_output_shapes
:���������*
T0
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:���������
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_3/weights3*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_3/weights3/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_3/weights3*
valueB
 *׳]�
�
/layer_3/weights3/Initializer/random_uniform/maxConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_3/weights3*
seed2 
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
layer_3/weights3
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_3/weights3/readIdentitylayer_3/weights3*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
layer_3/biases3
VariableV2*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_3/biases3/readIdentitylayer_3/biases3*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*
T0*'
_output_shapes
:���������
S
layer_3/ReluRelulayer_3/add*
T0*'
_output_shapes
:���������
�
0output/weights4/Initializer/random_uniform/shapeConst*"
_class
loc:@output/weights4*
valueB"      *
dtype0*
_output_shapes
:
�
.output/weights4/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@output/weights4*
valueB
 *qĜ�
�
.output/weights4/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@output/weights4*
valueB
 *qĜ?
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:*

seed *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
output/weights4
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container 
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
~
output/weights4/readIdentityoutput/weights4*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
output/biases4
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
w
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
g

output/addAddoutput/MatMuloutput/biases4/read*
T0*'
_output_shapes
:���������
s
cost/PlaceholderPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*
T0*'
_output_shapes
:���������
[

cost/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
_output_shapes
:*
T0*
out_type0
i
&train/gradients/cost/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
n
$train/gradients/cost/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
&train/gradients/cost/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*
T0*'
_output_shapes
:���������
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
T0*
out_type0*
_output_shapes
:
q
'train/gradients/output/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
_output_shapes
:*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul*'
_output_shapes
:���������
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_3/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*'
_output_shapes
:���������*
T0
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
train/beta1_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: *
dtype0*
_output_shapes
: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta1_power/readIdentitytrain/beta1_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam_1
VariableV2*#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam_1
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam
VariableV2*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_2/weights2*
valueB*    
�
train/layer_2/weights2/Adam_1
VariableV2*#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
�
train/layer_2/biases2/Adam
VariableV2*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container 
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
train/layer_3/biases3/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container 
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
train/layer_3/biases3/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
,train/output/weights4/Adam/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container 
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container 
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
+train/output/biases4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
�
-train/output/biases4/Adam_1/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam_1
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
]
train/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_1/weights1
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_2/weights2
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_2/biases2*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_3/biases3*
use_nesterov( *
_output_shapes
:
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*"
_class
loc:@output/weights4
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@output/biases4*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*"
_class
loc:@layer_1/biases1
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*"
_class
loc:@layer_1/biases1
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*%
valueB Blogging/current_cost*
dtype0*
_output_shapes
: 
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
T0*
_output_shapes
: 
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
�
save/SaveV2/tensor_namesConst*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign""�
trainable_variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08"'
	summaries

logging/current_cost:0"
train_op


train/Adam"�
	variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
�
train/layer_1/weights1/Adam:0"train/layer_1/weights1/Adam/Assign"train/layer_1/weights1/Adam/read:02/train/layer_1/weights1/Adam/Initializer/zeros:0
�
train/layer_1/weights1/Adam_1:0$train/layer_1/weights1/Adam_1/Assign$train/layer_1/weights1/Adam_1/read:021train/layer_1/weights1/Adam_1/Initializer/zeros:0
�
train/layer_1/biases1/Adam:0!train/layer_1/biases1/Adam/Assign!train/layer_1/biases1/Adam/read:02.train/layer_1/biases1/Adam/Initializer/zeros:0
�
train/layer_1/biases1/Adam_1:0#train/layer_1/biases1/Adam_1/Assign#train/layer_1/biases1/Adam_1/read:020train/layer_1/biases1/Adam_1/Initializer/zeros:0
�
train/layer_2/weights2/Adam:0"train/layer_2/weights2/Adam/Assign"train/layer_2/weights2/Adam/read:02/train/layer_2/weights2/Adam/Initializer/zeros:0
�
train/layer_2/weights2/Adam_1:0$train/layer_2/weights2/Adam_1/Assign$train/layer_2/weights2/Adam_1/read:021train/layer_2/weights2/Adam_1/Initializer/zeros:0
�
train/layer_2/biases2/Adam:0!train/layer_2/biases2/Adam/Assign!train/layer_2/biases2/Adam/read:02.train/layer_2/biases2/Adam/Initializer/zeros:0
�
train/layer_2/biases2/Adam_1:0#train/layer_2/biases2/Adam_1/Assign#train/layer_2/biases2/Adam_1/read:020train/layer_2/biases2/Adam_1/Initializer/zeros:0
�
train/layer_3/weights3/Adam:0"train/layer_3/weights3/Adam/Assign"train/layer_3/weights3/Adam/read:02/train/layer_3/weights3/Adam/Initializer/zeros:0
�
train/layer_3/weights3/Adam_1:0$train/layer_3/weights3/Adam_1/Assign$train/layer_3/weights3/Adam_1/read:021train/layer_3/weights3/Adam_1/Initializer/zeros:0
�
train/layer_3/biases3/Adam:0!train/layer_3/biases3/Adam/Assign!train/layer_3/biases3/Adam/read:02.train/layer_3/biases3/Adam/Initializer/zeros:0
�
train/layer_3/biases3/Adam_1:0#train/layer_3/biases3/Adam_1/Assign#train/layer_3/biases3/Adam_1/read:020train/layer_3/biases3/Adam_1/Initializer/zeros:0
�
train/output/weights4/Adam:0!train/output/weights4/Adam/Assign!train/output/weights4/Adam/read:02.train/output/weights4/Adam/Initializer/zeros:0
�
train/output/weights4/Adam_1:0#train/output/weights4/Adam_1/Assign#train/output/weights4/Adam_1/read:020train/output/weights4/Adam_1/Initializer/zeros:0
�
train/output/biases4/Adam:0 train/output/biases4/Adam/Assign train/output/biases4/Adam/read:02-train/output/biases4/Adam/Initializer/zeros:0
�
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign"train/output/biases4/Adam_1/read:02/train/output/biases4/Adam_1/Initializer/zeros:0w���(       �pJ	�g	��A*

logging/current_cost��=?��@*       ����	�	��A*

logging/current_costj��<���A*       ����	$�	��A
*

logging/current_cost��<�<�,*       ����	R	
��A*

logging/current_cost/��<�;@�*       ����	+6
��A*

logging/current_costY=�<�7�X*       ����	�g
��A*

logging/current_cost@��<
��[*       ����	�
��A*

logging/current_cost:^�<�^<
*       ����	�
��A#*

logging/current_costZX�<|��*       ����	:�
��A(*

logging/current_costn�<V
;*       ����	�+��A-*

logging/current_cost}1�<����*       ����	�]��A2*

logging/current_cost�O�<k�h-*       ����	 ���A7*

logging/current_cost֎�<��:.*       ����	���A<*

logging/current_cost�ɳ<#	�@*       ����	����AA*

logging/current_cost���<�*x�*       ����	$��AF*

logging/current_costC
�<��+*       ����	�D��AK*

logging/current_cost���<�c�*       ����	r��AP*

logging/current_cost�<G�7�*       ����	���AU*

logging/current_costB7�<]�9�*       ����	����AZ*

logging/current_costWd�<�H��*       ����	R���A_*

logging/current_cost�X�<�ۈ*       ����	�'��Ad*

logging/current_costN��<�"�*       ����	OV��Ai*

logging/current_cost�w�<����*       ����	>���An*

logging/current_cost!%�<�ݩ*       ����	���As*

logging/current_cost+I�<����*       ����	k���Ax*

logging/current_cost�=�<i��*       ����	N��A}*

logging/current_cost��z<h<Q+       ��K	t;��A�*

logging/current_cost�o<C[�`+       ��K	Ti��A�*

logging/current_cost�:c<S��+       ��K	����A�*

logging/current_cost+�W<+�T�+       ��K	b���A�*

logging/current_costx L<�b��+       ��K	p���A�*

logging/current_cost)�@<�{��+       ��K	5��A�*

logging/current_cost	�5<�l%+       ��K	hJ��A�*

logging/current_cost
�*<<�f�+       ��K	�x��A�*

logging/current_costT <i��+       ��K	����A�*

logging/current_costK�<��^+       ��K	C���A�*

logging/current_costZ?<���+       ��K	� ��A�*

logging/current_cost�K<ɰ�Y+       ��K	�.��A�*

logging/current_cost1��;U+       ��K	�]��A�*

logging/current_cost��;�ul+       ��K	���A�*

logging/current_cost��;<w+       ��K	b���A�*

logging/current_cost��;�R�R+       ��K	B���A�*

logging/current_cost-��;�j�g+       ��K	 ��A�*

logging/current_costb�;���+       ��K	\I��A�*

logging/current_cost~�;�Q+       ��K	�w��A�*

logging/current_cost�3�;
P��+       ��K	���A�*

logging/current_cost��;l��+       ��K	[���A�*

logging/current_costo��;H���+       ��K	���A�*

logging/current_costYۗ;_؞�+       ��K	#P��A�*

logging/current_costY��;��>�+       ��K	׃��A�*

logging/current_cost��;���^+       ��K	����A�*

logging/current_cost|ُ;�黋+       ��K	� ��A�*

logging/current_costE�;�6�T+       ��K	$?��A�*

logging/current_cost��;̟+       ��K	���A�*

logging/current_costV�;c��+       ��K	����A�*

logging/current_cost�W�;[���+       ��K	����A�*

logging/current_cost�;�FL~+       ��K	'/��A�*

logging/current_cost�;�;^=B+       ��K	m��A�*

logging/current_costC͉;~lrN+       ��K	����A�*

logging/current_costRn�;´�J+       ��K	����A�*

logging/current_cost<�;�s�+       ��K	���A�*

logging/current_cost�ʈ;ѿ��+       ��K	?T��A�*

logging/current_cost���;��;�+       ��K	0���A�*

logging/current_cost<H�;�n~�+       ��K	j���A�*

logging/current_cost��;1���+       ��K	���A�*

logging/current_costۇ;�ks+       ��K	'��A�*

logging/current_cost/��; ��+       ��K	c��A�*

logging/current_cost���;n�p+       ��K	���A�*

logging/current_cost�t�;xe�Y+       ��K	.���A�*

logging/current_cost]�;�m�f+       ��K	���A�*

logging/current_cost�N�;���F+       ��K	#-��A�*

logging/current_cost�J�;N�F�+       ��K	�Y��A�*

logging/current_cost�G�;/�+       ��K	����A�*

logging/current_cost�D�;�d��+       ��K	\���A�*

logging/current_costMB�;מz+       ��K	����A�*

logging/current_cost�?�;׹�+       ��K	J'��A�*

logging/current_costN=�;�x5)+       ��K	�Y��A�*

logging/current_cost';�;���+       ��K	���A�*

logging/current_costw9�;���+       ��K	W���A�*

logging/current_cost78�;���=+       ��K	e��A�*

logging/current_cost7�;�_V�+       ��K	�8��A�*

logging/current_cost:6�;�j�+       ��K	`u��A�*

logging/current_costt5�;[�+       ��K	צ��A�*

logging/current_cost�4�;�8Q�+       ��K	����A�*

logging/current_cost�3�;L���+       ��K	"��A�*

logging/current_cost'3�;]%'�+       ��K	r>��A�*

logging/current_cost<2�;�&�+       ��K	Tm��A�*

logging/current_costM1�;�m(+       ��K	����A�*

logging/current_cost�0�;�u�x+       ��K	A���A�*

logging/current_cost�/�;�� +       ��K	m���A�*

logging/current_cost8/�;��+       ��K	W.��A�*

logging/current_cost�.�;7	܂+       ��K	�]��A�*

logging/current_cost.�;&4�+       ��K	����A�*

logging/current_costl-�;��b�+       ��K	v���A�*

logging/current_cost�,�;G��T+       ��K	����A�*

logging/current_cost*,�;['y�+       ��K	S��A�*

logging/current_cost�+�;���+       ��K	�M��A�*

logging/current_cost
+�;Y��V+       ��K	T���A�*

logging/current_cost�*�;)�)�+       ��K	���A�*

logging/current_cost�)�;g��+       ��K	����A�*

logging/current_cost�)�;��~+       ��K	���A�*

logging/current_cost�(�;�I�+       ��K	mA��A�*

logging/current_cost=(�;%QB�+       ��K	�r��A�*

logging/current_cost�'�;����+       ��K	%���A�*

logging/current_cost'�;j�`z+       ��K	����A�*

logging/current_cost�&�;r�+�+       ��K	>���A�*

logging/current_cost�%�;�XUf+       ��K	6,��A�*

logging/current_costN%�;�i�+       ��K	�]��A�*

logging/current_cost�$�;F��5+       ��K	����A�*

logging/current_cost1$�;׀��+       ��K	���A�*

logging/current_cost�#�;?��+       ��K	D���A�*

logging/current_cost#�;d_+       ��K	�!��A�*

logging/current_cost]"�;~7d{+       ��K	T��A�*

logging/current_cost�!�;�S�f+       ��K	n���A�*

logging/current_cost[!�;H6�+       ��K	ñ��A�*

logging/current_cost� �;b0b+       ��K	S���A�*

logging/current_cost+ �;��+       ��K	6 ��A�*

logging/current_cost��;�A+       ��K	E= ��A�*

logging/current_cost��;���x+       ��K	fo ��A�*

logging/current_costc�;�c�o+       ��K	ȝ ��A�*

logging/current_cost��;��+       ��K	&� ��A�*

logging/current_cost9�;�;+       ��K	{� ��A�*

logging/current_cost��;ِ�+       ��K	'!��A�*

logging/current_cost[�;��P+       ��K	6V!��A�*

logging/current_cost��;� 
+       ��K	�!��A�*

logging/current_cost�;"�F�+       ��K	}�!��A�*

logging/current_cost��;��[+       ��K	�!��A�*

logging/current_cost�;tm�m+       ��K	�"��A�*

logging/current_cost��;�Ԡ�+       ��K	U>"��A�*

logging/current_cost7�;<o��+       ��K	m"��A�*

logging/current_cost��;���9+       ��K	�"��A�*

logging/current_cost%�;9h^�+       ��K	��"��A�*

logging/current_costm�;P��+       ��K	x�"��A�*

logging/current_cost��;�ܗ+       ��K	,&#��A�*

logging/current_cost3�;�e0+       ��K	�X#��A�*

logging/current_cost��;c�O.+       ��K	��#��A�*

logging/current_cost@�;����+       ��K	׼#��A�*

logging/current_cost��;\D7�+       ��K	�#��A�*

logging/current_cost��;�/��+       ��K	2$��A�*

logging/current_cost��;�Eڷ+       ��K	�K$��A�*

logging/current_cost�;r�*y+       ��K	y$��A�*

logging/current_cost��;���?+       ��K	i�$��A�*

logging/current_cost��;��C+       ��K	��$��A�*

logging/current_costP�;f��?+       ��K	u%��A�*

logging/current_cost��;/(1�+       ��K	�1%��A�*

logging/current_cost/�;���+       ��K	�_%��A�*

logging/current_cost��;Q:T+       ��K	y�%��A�*

logging/current_cost�;��=X+       ��K	��%��A�*

logging/current_cost��;���/+       ��K	�%��A�*

logging/current_cost�;���M+       ��K	$#&��A�*

logging/current_cost��;���+       ��K	U&��A�*

logging/current_cost��; �?+       ��K	V�&��A�*

logging/current_cost�;U���+       ��K	�&��A�*

logging/current_cost��;t��f+       ��K	��&��A�*

logging/current_costt�;���+       ��K	�'��A�*

logging/current_cost�
�;z
�l+       ��K	�J'��A�*

logging/current_costW
�;�r�+       ��K	9y'��A�*

logging/current_cost�	�;���+       ��K	q�'��A�*

logging/current_cost7	�;��+       ��K		�'��A�*

logging/current_cost��;a�$+       ��K	M(��A�*

logging/current_cost�;d��t+       ��K	|<(��A�*

logging/current_cost��;��+       ��K	j(��A�*

logging/current_costC�;�	�+       ��K	d�(��A�*

logging/current_cost��;33�6+       ��K	Q�(��A�*

logging/current_costL�;肔�+       ��K	d�(��A�*

logging/current_cost��;�h�+       ��K	I()��A�*

logging/current_cost��; �ku+       ��K	�U)��A�*

logging/current_cost��;��+       ��K	z�)��A�*

logging/current_cost'�;���q+       ��K	�)��A�*

logging/current_cost�;.�K1+       ��K	��)��A�*

logging/current_cost��;�'+{+       ��K	�*��A�*

logging/current_costw�;�Rw+       ��K	�?*��A�*

logging/current_cost�;�rf�+       ��K	gl*��A�*

logging/current_costc�;��
+       ��K	�*��A�*

logging/current_cost� �;��E+       ��K	��*��A�*

logging/current_cost �;ȣ�p+       ��K	��*��A�*

logging/current_costq��;9䰠+       ��K	�&+��A�*

logging/current_cost���;^�	+       ��K	�U+��A�*

logging/current_costf��;�{p�+       ��K	��+��A�*

logging/current_cost���;�	T+       ��K	��+��A�*

logging/current_cost0��;:Ą,+       ��K	��+��A�*

logging/current_cost���;�_�+       ��K	�,��A�*

logging/current_cost!��;��P�+       ��K	k:,��A�*

logging/current_cost���;���h+       ��K	�h,��A�*

logging/current_cost��;g�|+       ��K	��,��A�*

logging/current_costd��;,�$�+       ��K	a�,��A�*

logging/current_cost���;"���+       ��K	��,��A�*

logging/current_costO��;d��=+       ��K	#-��A�*

logging/current_cost���;�+       ��K	#S-��A�*

logging/current_costz��;����+       ��K	��-��A�*

logging/current_cost���;���+       ��K	��-��A�*

logging/current_costM��;�[�+       ��K	��-��A�*

logging/current_cost���;Gը+       ��K	
.��A�*

logging/current_cost��;�4�}+       ��K	%7.��A�*

logging/current_cost���;�%l +       ��K	�e.��A�*

logging/current_cost��;��+       ��K	%�.��A�*

logging/current_cost��; �&+       ��K	��.��A�*

logging/current_cost0�;����+       ��K	V�.��A�*

logging/current_costr�;���+       ��K	�/��A�*

logging/current_cost��;܍�+       ��K	CF/��A�*

logging/current_cost>�;�:�Q+       ��K	v/��A�*

logging/current_cost��;��+       ��K	��/��A�*

logging/current_cost1�;̠-/+       ��K	:�/��A�*

logging/current_cost���;
2i�+       ��K	C0��A�*

logging/current_cost"��;cš�+       ��K	�20��A�*

logging/current_cost��;J��+       ��K	�h0��A�*

logging/current_cost�;!-�+       ��K	|�0��A�*

logging/current_costv�;���+       ��K	��0��A�*

logging/current_cost �;��+       ��K	��0��A�*

logging/current_coste�;)b��+       ��K	�(1��A�*

logging/current_costD�;�D�T+       ��K	Z1��A�*

logging/current_cost��;��)�+       ��K	-�1��A�*

logging/current_cost*�;�lH+       ��K	L�1��A�*

logging/current_cost��;{e��+       ��K	��1��A�*

logging/current_cost�;���+       ��K	�2��A�*

logging/current_costm�;�R(+       ��K	~D2��A�*

logging/current_cost��;���+       ��K	�q2��A�*

logging/current_costQ�;�B��+       ��K	 �2��A�*

logging/current_cost��;�m7s+       ��K	��2��A�*

logging/current_cost��;g��+       ��K	��2��A�*

logging/current_cost��;�j�+       ��K	M-3��A�*

logging/current_costR�;�9�h+       ��K	�[3��A�*

logging/current_cost��;M�Z=+       ��K	��3��A�*

logging/current_costP�;���+       ��K	V�3��A�*

logging/current_cost��;�8!A+       ��K	8�3��A�*

logging/current_cost@�;"�+       ��K	�4��A�*

logging/current_cost��;7Db�+       ��K	:J4��A�*

logging/current_cost@�;�Η-+       ��K	�w4��A�*

logging/current_cost��;1а8+       ��K	��4��A�*

logging/current_costI�;p�u+       ��K	v�4��A�*

logging/current_cost��;:��+       ��K	b5��A�*

logging/current_costG�;�s9+       ��K	r25��A�	*

logging/current_cost��;5&�+       ��K	wa5��A�	*

logging/current_costO�;X<��+       ��K	Đ5��A�	*

logging/current_cost���;؊�d+       ��K	�5��A�	*

logging/current_costQ��;�r��+       ��K	�5��A�	*

logging/current_cost�߆;$}K*+       ��K	!6��A�	*

logging/current_costQ߆;��ݝ+       ��K	�H6��A�	*

logging/current_cost�ކ;l��R+       ��K	x6��A�	*

logging/current_costLކ;���c+       ��K	B�6��A�	*

logging/current_cost�݆;�[l�+       ��K	��6��A�	*

logging/current_costQ݆;���	+       ��K	�7��A�	*

logging/current_cost�܆;Ss�+       ��K	?17��A�	*

logging/current_cost\܆;��+       ��K	_^7��A�	*

logging/current_cost�ۆ;�U��+       ��K	"�7��A�	*

logging/current_cost_ۆ;��͛+       ��K	�7��A�	*

logging/current_cost�چ;u]	+       ��K	1�7��A�	*

logging/current_costnچ;��G+       ��K	�"8��A�	*

logging/current_costچ;( m-+       ��K	�Q8��A�	*

logging/current_costyن;['ݞ+       ��K	�8��A�	*

logging/current_costن;n�Iq+       ��K	İ8��A�	*

logging/current_cost�؆;���%+       ��K	j�8��A�	*

logging/current_cost؆;�UlP+       ��K	I
9��A�	*

logging/current_cost�׆;#X��+       ��K	:9��A�	*

logging/current_costQ׆;�q�L+       ��K	Ek9��A�	*

logging/current_cost�ֆ;8��+       ��K	��9��A�	*

logging/current_cost)ֆ;=ifz+       ��K	%�9��A�
*

logging/current_cost�Ն;��16+       ��K	:��A�
*

logging/current_costBՆ;W��+       ��K	/5:��A�
*

logging/current_cost�Ԇ;��3+       ��K	#c:��A�
*

logging/current_costLԆ;����+       ��K	��:��A�
*

logging/current_cost�ӆ;�n��+       ��K	˾:��A�
*

logging/current_costjӆ;�%�'+       ��K	y�:��A�
*

logging/current_cost�҆;��+       ��K	;��A�
*

logging/current_costr҆;��l+       ��K	@L;��A�
*

logging/current_cost�ц;j�#l+       ��K	3�;��A�
*

logging/current_costrц;��"l+       ��K	��;��A�
*

logging/current_cost*ц;��+       ��K	I�;��A�
*

logging/current_costkІ; 2�+       ��K	�+<��A�
*

logging/current_cost�φ;�}*�+       ��K	�X<��A�
*

logging/current_cost�Ά;ī+       ��K	9�<��A�
*

logging/current_cost(͆;��@+       ��K	��<��A�
*

logging/current_cost�ˆ;��m+       ��K	f�<��A�
*

logging/current_cost5ʆ;%��+       ��K	�=��A�
*

logging/current_cost�Ȇ;�zN�+       ��K	Y==��A�
*

logging/current_cost�ǆ;�AM+       ��K	8s=��A�
*

logging/current_cost�Ɔ;�-(�+       ��K	,�=��A�
*

logging/current_cost*Ɔ;`���+       ��K	i�=��A�
*

logging/current_costPņ;��W�+       ��K	�>��A�
*

logging/current_costUĆ;a�c+       ��K	�3>��A�
*

logging/current_cost�Æ;�{�+       ��K	�`>��A�
*

logging/current_cost�;5yd{+       ��K	)�>��A�
*

logging/current_cost5;+��+       ��K	�>��A�*

logging/current_costY��;�)�+       ��K	6�>��A�*

logging/current_cost���;���\+       ��K	1?��A�*

logging/current_cost�;�p�+       ��K	yJ?��A�*

logging/current_cost$��;��+       ��K	Ny?��A�*

logging/current_costv��;~��+       ��K	t�?��A�*

logging/current_cost���;M�.+       ��K	��?��A�*

logging/current_cost���;�O��+       ��K	�@��A�*

logging/current_costH��;����+       ��K	4@��A�*

logging/current_cost���;n)x +       ��K	c@��A�*

logging/current_costϺ�;$�+�+       ��K	��@��A�*

logging/current_cost7��;�ݦ+       ��K	0�@��A�*

logging/current_costz��;T��+       ��K	7�@��A�*

logging/current_cost���;�>�s+       ��K	�)A��A�*

logging/current_cost��;�@+       ��K	�XA��A�*

logging/current_costf��;K}J+       ��K	�A��A�*

logging/current_cost���;!��+       ��K	ɵA��A�*

logging/current_cost���;��
-+       ��K	{�A��A�*

logging/current_costQ��;�<j+       ��K	�B��A�*

logging/current_cost���;#N� +       ��K	�BB��A�*

logging/current_cost���;D��y+       ��K	SqB��A�*

logging/current_cost4��;O��+       ��K	y�B��A�*

logging/current_cost���;�O�k+       ��K	�B��A�*

logging/current_cost屆;]��+       ��K	�C��A�*

logging/current_cost ��;	vx+       ��K	v>C��A�*

logging/current_cost���;�]+       ��K	oC��A�*

logging/current_costெ;rO�A+       ��K	�C��A�*

logging/current_cost)��;�+W{+       ��K	Y�C��A�*

logging/current_cost���;�ܢ"+       ��K	H�C��A�*

logging/current_costϭ�;;(�j+       ��K	p&D��A�*

logging/current_cost!��;�h�+       ��K	�UD��A�*

logging/current_costf��;�� +       ��K	p�D��A�*

logging/current_cost嫆;�B�+       ��K	��D��A�*

logging/current_cost��;�P��+       ��K	��D��A�*

logging/current_cost���;�g�*+       ��K	[E��A�*

logging/current_costᩆ;��+       ��K	<E��A�*

logging/current_costI��;�8E�+       ��K	�jE��A�*

logging/current_cost~��;��g+       ��K	��E��A�*

logging/current_cost���;vZo�+       ��K	��E��A�*

logging/current_costw��;Y	�+       ��K	��E��A�*

logging/current_cost���;�e3�+       ��K	� F��A�*

logging/current_cost���;�Gż+       ��K	NF��A�*

logging/current_costu��;ru��+       ��K	�|F��A�*

logging/current_cost���;zVn+       ��K	�F��A�*

logging/current_cost=��;��F�+       ��K	��F��A�*

logging/current_cost~��;w�x+       ��K	�G��A�*

logging/current_cost��;T��H+       ��K	x3G��A�*

logging/current_costS��;��o+       ��K	�`G��A�*

logging/current_cost|��;���+       ��K	>�G��A�*

logging/current_cost順;ț�=+       ��K	L�G��A�*

logging/current_costD��;Ҟ�n+       ��K	��G��A�*

logging/current_costȟ�;��X+       ��K	rH��A�*

logging/current_cost/��;��I�+       ��K	3FH��A�*

logging/current_costr��;t�A$+       ��K	;rH��A�*

logging/current_cost흆;;*�+       ��K	&�H��A�*

logging/current_costP��;��y�+       ��K	/�H��A�*

logging/current_cost���;�e�+       ��K	8�H��A�*

logging/current_cost<��;;a��+       ��K	T(I��A�*

logging/current_cost���;ˀ;+       ��K	�VI��A�*

logging/current_cost���;�y�+       ��K	لI��A�*

logging/current_cost���;�د+       ��K	�I��A�*

logging/current_cost癆;�
�+       ��K	��I��A�*

logging/current_costb��;�_{+       ��K	�J��A�*

logging/current_costӘ�;�s+       ��K	�=J��A�*

logging/current_costC��;�k@d+       ��K	ssJ��A�*

logging/current_cost���;���i+       ��K	x�J��A�*

logging/current_cost#��;�ҩ+       ��K	F�J��A�*

logging/current_cost���;���+       ��K	�K��A�*

logging/current_cost��;Q&��+       ��K	�/K��A�*

logging/current_costZ��;\_+       ��K	-\K��A�*

logging/current_costؔ�;�z+       ��K	�K��A�*

logging/current_costR��;��~�+       ��K	�K��A�*

logging/current_cost���;���+       ��K	L��A�*

logging/current_cost��;�>��+       ��K	yLL��A�*

logging/current_cost���;�G�+       ��K	��L��A�*

logging/current_cost-��;Nkl+       ��K	o�L��A�*

logging/current_cost���;�E�+       ��K	�M��A�*

logging/current_cost���;֮�+       ��K	#RM��A�*

logging/current_cost���;�һy+       ��K	ȒM��A�*

logging/current_cost��;���+       ��K	F�M��A�*

logging/current_cost���;T���+       ��K	w N��A�*

logging/current_cost���;�4�{+       ��K	�:N��A�*

logging/current_cost`��;�m�+       ��K	�qN��A�*

logging/current_cost;�b)�+       ��K	��N��A�*

logging/current_costr��;N!�}+       ��K	��N��A�*

logging/current_cost쌆;[�8+       ��K	�O��A�*

logging/current_costh��;m���+       ��K	$SO��A�*

logging/current_cost݋�;J9_�+       ��K	#�O��A�*

logging/current_costZ��;�{/K+       ��K	}�O��A�*

logging/current_cost���;���+       ��K	��O��A�*

logging/current_cost���;U�R+       ��K	Y(P��A�*

logging/current_cost��;9�D�+       ��K	�[P��A�*

logging/current_cost���;F�+       ��K	I�P��A�*

logging/current_cost
��;a�kf+       ��K	j�P��A�*

logging/current_cost���;Hl^�+       ��K	��P��A�*

logging/current_cost��;����+       ��K	�)Q��A�*

logging/current_cost���;�֟�+       ��K	�ZQ��A�*

logging/current_cost ��;3�G+       ��K	�Q��A�*

logging/current_cost���;&�ԑ+       ��K	8�Q��A�*

logging/current_costC��;|�+       ��K	��Q��A�*

logging/current_costȅ�;TM��+       ��K	� R��A�*

logging/current_costA��;	"�+       ��K	�YR��A�*

logging/current_cost���;�`.+       ��K	��R��A�*

logging/current_cost4��;�V+       ��K	m�R��A�*

logging/current_cost���;X��+       ��K	��R��A�*

logging/current_cost1��;A1"H+       ��K	�S��A�*

logging/current_cost���;j��c+       ��K	UQS��A�*

logging/current_cost=��;���+       ��K	sS��A�*

logging/current_costՁ�;�>T>+       ��K	ȰS��A�*

logging/current_costj��;	Z+       ��K	��S��A�*

logging/current_cost���;bc� +       ��K	T��A�*

logging/current_cost��;�}F+       ��K	#AT��A�*

logging/current_cost��;��i+       ��K	2pT��A�*

logging/current_cost��;�+��+       ��K	ݝT��A�*

logging/current_cost�;gK��+       ��K	/�T��A�*

logging/current_cost�~�;�"+       ��K	��T��A�*

logging/current_cost5~�;st�+       ��K	6)U��A�*

logging/current_cost�}�;/�ɖ+       ��K	�VU��A�*

logging/current_costY}�;�%'+       ��K	��U��A�*

logging/current_cost�|�;��8+       ��K	4�U��A�*

logging/current_cost�|�;L)+       ��K	X�U��A�*

logging/current_cost|�;(�'+       ��K	�V��A�*

logging/current_cost�{�;Q'H+       ��K	�8V��A�*

logging/current_costB{�;��G�+       ��K	fV��A�*

logging/current_cost�z�;l.��+       ��K	�V��A�*

logging/current_costrz�;w���+       ��K	>�V��A�*

logging/current_costz�;��x+       ��K	��V��A�*

logging/current_cost�y�;���7+       ��K	�W��A�*

logging/current_cost9y�;�;`+       ��K	�HW��A�*

logging/current_cost�x�;\�9+       ��K	�|W��A�*

logging/current_costlx�;wb��+       ��K	�W��A�*

logging/current_costx�;�K'�+       ��K	��W��A�*

logging/current_cost�w�;b�Sj+       ��K	X��A�*

logging/current_cost@w�;D�9�+       ��K	�GX��A�*

logging/current_cost�v�;_�q�+       ��K	|X��A�*

logging/current_cost~v�;Z��+       ��K	�X��A�*

logging/current_costv�;�<�+       ��K	�X��A�*

logging/current_cost�u�;ihV�+       ��K	mY��A�*

logging/current_costau�;-؍:+       ��K	�EY��A�*

logging/current_cost u�;�Z}+       ��K	]sY��A�*

logging/current_cost�t�;����+       ��K	��Y��A�*

logging/current_cost?t�;Rz��+       ��K	��Y��A�*

logging/current_cost�s�;�,�+       ��K	Z��A�*

logging/current_cost�s�;W�#A+       ��K	�7Z��A�*

logging/current_cost1s�;��x+       ��K		hZ��A�*

logging/current_cost�r�;���+       ��K	J�Z��A�*

logging/current_costr�;��+       ��K	*�Z��A�*

logging/current_cost!r�;j��Z+       ��K	��Z��A�*

logging/current_cost�q�;��t�+       ��K	\,[��A�*

logging/current_costqq�;��!�+       ��K	]e[��A�*

logging/current_costq�;m��+       ��K	��[��A�*

logging/current_cost�p�;3t�+       ��K	�[��A�*

logging/current_costnp�;���+       ��K	��[��A�*

logging/current_costp�;ţ�+       ��K	2\��A�*

logging/current_cost�o�;�!�+       ��K	�O\��A�*

logging/current_costso�;@/+       ��K	!�\��A�*

logging/current_cost!o�;Х�++       ��K	)�\��A�*

logging/current_cost�n�;�i�H+       ��K	:�\��A�*

logging/current_cost~n�;�^D�+       ��K	�]��A�*

logging/current_cost-n�;�4�+       ��K	D]��A�*

logging/current_cost�m�;`f�+       ��K	tt]��A�*

logging/current_cost�m�;�.+       ��K	(�]��A�*

logging/current_cost?m�;Yn�+       ��K	"�]��A�*

logging/current_cost�l�;�|�/+       ��K	F�]��A�*

logging/current_cost�l�;���y+       ��K	 /^��A�*

logging/current_costVl�;<�T+       ��K	s\^��A�*

logging/current_costl�;z�+       ��K	8�^��A�*

logging/current_cost�k�;I�V+       ��K	��^��A�*

logging/current_costjk�;q�2�+       ��K	x�^��A�*

logging/current_costk�;��l�+       ��K	D_��A�*

logging/current_cost�j�;�kA?+       ��K	r?_��A�*

logging/current_cost�j�;/�(�+       ��K	�l_��A�*

logging/current_cost=j�;'��+       ��K	�_��A�*

logging/current_cost�i�;�2+       ��K		�_��A�*

logging/current_cost�i�;��BW+       ��K	�_��A�*

logging/current_cost^i�; ��+       ��K	�&`��A�*

logging/current_costi�;	M�6+       ��K	�U`��A�*

logging/current_cost�h�;E>R<+       ��K	�`��A�*

logging/current_cost�h�;�~��+       ��K	�`��A�*

logging/current_cost7h�;H���+       ��K	!�`��A�*

logging/current_cost�g�;���+       ��K	�a��A�*

logging/current_cost�g�;7jl�+       ��K	g:a��A�*

logging/current_costag�;�r\�+       ��K	�ea��A�*

logging/current_costg�;�z&)+       ��K	��a��A�*

logging/current_cost�f�;��Y+       ��K	u�a��A�*

logging/current_cost�f�;��;+       ��K	�a��A�*

logging/current_costFf�;���+       ��K	�b��A�*

logging/current_costf�;�r�+       ��K	�Gb��A�*

logging/current_cost�e�;5��f+       ��K	�ub��A�*

logging/current_cost{e�;��*I+       ��K	�b��A�*

logging/current_cost:e�;��r6+       ��K	$�b��A�*

logging/current_cost�d�;_�j�+       ��K	� c��A�*

logging/current_cost�d�;	���+       ��K	s/c��A�*

logging/current_costrd�;��[+       ��K	�[c��A�*

logging/current_cost4d�;56+       ��K	��c��A�*

logging/current_cost�c�;�L5�+       ��K	ߵc��A�*

logging/current_cost�c�;dj�+       ��K	��c��A�*

logging/current_costGc�;x�x1+       ��K	Wd��A�*

logging/current_cost�b�;E\k�+       ��K	fBd��A�*

logging/current_cost�b�;|W+       ��K	�sd��A�*

logging/current_costcb�;'�+       ��K	��d��A�*

logging/current_costb�;UKtt+       ��K	��d��A�*

logging/current_cost�a�;��s+       ��K	��d��A�*

logging/current_cost�a�;X�_+       ��K	]/e��A�*

logging/current_cost@a�;C�!H+       ��K	�\e��A�*

logging/current_cost�`�;�uXm+       ��K	��e��A�*

logging/current_cost�`�;�7��+       ��K	�e��A�*

logging/current_costo`�;U�	�+       ��K	��e��A�*

logging/current_cost)`�;��$+       ��K	�$f��A�*

logging/current_cost�_�;����+       ��K	�Rf��A�*

logging/current_cost�_�;؁&�+       ��K	��f��A�*

logging/current_costd_�;*�a�+       ��K	̲f��A�*

logging/current_cost#_�;`f�8+       ��K	o�f��A�*

logging/current_cost�^�;b^�+       ��K	�g��A�*

logging/current_cost�^�;-���+       ��K	gFg��A�*

logging/current_costc^�;|��v+       ��K	�xg��A�*

logging/current_cost$^�;�PO�+       ��K	F�g��A�*

logging/current_cost�]�;���+       ��K	2�g��A�*

logging/current_cost�]�;�_A�+       ��K	�h��A�*

logging/current_costj]�;�ӋO+       ��K	<3h��A�*

logging/current_cost.]�;���+       ��K	N`h��A�*

logging/current_cost�\�;��
+       ��K	��h��A�*

logging/current_cost�\�;/��]+       ��K	G�h��A�*

logging/current_cost�\�;�[��+       ��K	��h��A�*

logging/current_costE\�;�zê+       ��K	h!i��A�*

logging/current_cost\�;
9=�+       ��K	Oi��A�*

logging/current_cost�[�;��?+       ��K	W�i��A�*

logging/current_cost�[�;c�*�+       ��K	S�i��A�*

logging/current_costb[�;�	�+       ��K	��i��A�*

logging/current_cost.[�;f�h�+       ��K	�j��A�*

logging/current_cost�Z�;[!�
+       ��K	�<j��A�*

logging/current_cost�Z�;�|��+       ��K	�kj��A�*

logging/current_cost�Z�;u�F_+       ��K	�j��A�*

logging/current_cost]Z�;����+       ��K	<�j��A�*

logging/current_cost*Z�;sh��+       ��K	l�j��A�*

logging/current_cost�Y�;���+       ��K	�&k��A�*

logging/current_cost�Y�;���+       ��K	Wk��A�*

logging/current_cost�Y�;���H+       ��K	��k��A�*

logging/current_costfY�;����+       ��K	6�k��A�*

logging/current_cost;Y�;g(��+       ��K	��k��A�*

logging/current_costY�;���O+       ��K	�l��A�*

logging/current_cost�X�;�ܦ+       ��K	�@l��A�*

logging/current_cost�X�;��Fl+       ��K	�nl��A�*

logging/current_cost�X�;�=��+       ��K	l��A�*

logging/current_costXX�;^4�}+       ��K	7�l��A�*

logging/current_cost.X�;K�d+       ��K	��l��A�*

logging/current_costX�;��=�+       ��K	4%m��A�*

logging/current_cost�W�;���+       ��K	hSm��A�*

logging/current_cost�W�;h��P+       ��K	��m��A�*

logging/current_cost�W�;�P)�+       ��K	j�m��A�*

logging/current_cost�V�;S��+       ��K	��m��A�*

logging/current_cost0V�;=+=�+       ��K	�n��A�*

logging/current_cost�T�;�﫺+       ��K	�;n��A�*

logging/current_cost�S�;���.+       ��K	Akn��A�*

logging/current_cost(Q�;/���+       ��K	C�n��A�*

logging/current_costL�;�93�+       ��K	4�n��A�*

logging/current_costA�;!�H +       ��K	��n��A�*

logging/current_cost�6�;5�x+       ��K	�"o��A�*

logging/current_cost�.�;<5Ə+       ��K	�Po��A�*

logging/current_cost�*�;l>��+       ��K	?o��A�*

logging/current_cost�%�;.�b+       ��K	�o��A�*

logging/current_cost��;m8T+       ��K	�o��A�*

logging/current_cost?�;*�1+       ��K	ap��A�*

logging/current_cost��;�?'+       ��K	�9p��A�*

logging/current_cost��;ЌW�+       ��K	�hp��A�*

logging/current_cost��;]z+       ��K	��p��A�*

logging/current_cost
�;��^+       ��K	��p��A�*

logging/current_cost�
�;��)�+       ��K	��p��A�*

logging/current_cost�;9+       ��K	� q��A�*

logging/current_cost��;TR�++       ��K	Sq��A�*

logging/current_cost��;i�q�+       ��K	L�q��A�*

logging/current_cost��;h���+       ��K	رq��A�*

logging/current_costh��;�O:�+       ��K	��q��A�*

logging/current_costq��;赯�+       ��K	�r��A�*

logging/current_costo��;v�� +       ��K	�@r��A�*

logging/current_cost���;& ё+       ��K	�or��A�*

logging/current_cost���;0���+       ��K	5�r��A�*

logging/current_cost���;̐�O+       ��K	��r��A�*

logging/current_cost��;�}��+       ��K	�s��A�*

logging/current_cost��;m��+       ��K	=2s��A�*

logging/current_cost��;�*e�+       ��K	_s��A�*

logging/current_cost �;�}n+       ��K		�s��A�*

logging/current_coste�;'C�+       ��K	e�s��A�*

logging/current_cost��;#pO�+       ��K	<�s��A�*

logging/current_costk�;BD'+       ��K	�"t��A�*

logging/current_cost�;��f+       ��K	�St��A�*

logging/current_cost�ޅ;�Y�+       ��K	��t��A�*

logging/current_cost/܅;�/l+       ��K	'�t��A�*

logging/current_costڅ;!�ٿ+       ��K	��t��A�*

logging/current_cost�ׅ;����+       ��K	�u��A�*

logging/current_cost�ԅ;hua+       ��K	�Bu��A�*

logging/current_costӅ;�.y+       ��K	�qu��A�*

logging/current_cost<Ѕ;�_��+       ��K	t�u��A�*

logging/current_cost�ͅ;�G� +       ��K	s�u��A�*

logging/current_cost̅;i�0F+       ��K	�v��A�*

logging/current_cost�ʅ;�DU+       ��K	�0v��A�*

logging/current_costqȅ;��3�+       ��K	�_v��A�*

logging/current_costƅ;­�+       ��K	��v��A�*

logging/current_cost�;S�x@+       ��K	�v��A�*

logging/current_cost���;B���+       ��K	~�v��A�*

logging/current_cost�;���+       ��K	�w��A�*

logging/current_cost)��;Q�b4+       ��K	Iw��A�*

logging/current_costܷ�;C���+       ��K	xw��A�*

logging/current_cost��;����+       ��K	ǥw��A�*

logging/current_cost���;����+       ��K	A�w��A�*

logging/current_cost��;�:�+       ��K	�x��A�*

logging/current_cost?��;�R�[+       ��K	^6x��A�*

logging/current_costѫ�;���m+       ��K	Vdx��A�*

logging/current_cost���;\;+       ��K	<�x��A�*

logging/current_costu��;���U+       ��K	�x��A�*

logging/current_cost���;,�:/+       ��K	n�x��A�*

logging/current_costl��;cO+       ��K	Xy��A�*

logging/current_cost��;�/J:+       ��K	4Iy��A�*

logging/current_costK��;	�q0+       ��K	�vy��A�*

logging/current_costw��;K�+       ��K	��y��A�*

logging/current_cost훅;>�+       ��K	��y��A�*

logging/current_cost♅;��K/+       ��K	c�y��A�*

logging/current_cost���;ve+       ��K	�,z��A�*

logging/current_cost���;C6��+       ��K	�[z��A�*

logging/current_cost���;J�2�+       ��K	��z��A�*

logging/current_cost���;#�^�+       ��K	��z��A�*

logging/current_costܑ�;��m+       ��K	��z��A�*

logging/current_cost���;2'�+       ��K	�{��A�*

logging/current_costX��;ҰO�+       ��K	�A{��A�*

logging/current_costQ��;|�/+       ��K	�{��A�*

logging/current_costT��;P��+       ��K	��{��A�*

logging/current_cost��;�u�+       ��K	�'|��A�*

logging/current_costU��;x�^+       ��K	�o|��A�*

logging/current_cost���;l�>+       ��K	��|��A�*

logging/current_costӅ�;:��+       ��K	��|��A�*

logging/current_costM��;_�5g+       ��K	�>}��A�*

logging/current_cost9��;\�35+       ��K	�t}��A�*

logging/current_cost`��;kث�+       ��K	֫}��A�*

logging/current_cost*��;2v�+       ��K	^�}��A�*

logging/current_cost��;�U��+       ��K	I~��A�*

logging/current_cost~�;pN+�+       ��K	dN~��A�*

logging/current_cost}�;����+       ��K	ʇ~��A�*

logging/current_cost!|�;F ͤ+       ��K	g�~��A�*

logging/current_cost�z�;]���+       ��K	��~��A�*

logging/current_cost�y�;�XL�+       ��K	+��A�*

logging/current_cost]y�;*@��+       ��K	c��A�*

logging/current_costGx�;���+       ��K	����A�*

logging/current_costRw�;�f�+       ��K	���A�*

logging/current_costUv�;��3*+       ��K	����A�*

logging/current_costJu�;��/G+       ��K	�/���A�*

logging/current_costKt�;�6+       ��K	�a���A�*

logging/current_costas�;�1�@+       ��K	ʒ���A�*

logging/current_cost�r�;��v+       ��K	�ǀ��A�*

logging/current_cost�q�;.,�U+       ��K	�����A�*

logging/current_costjp�;��� +       ��K	O*���A�*

logging/current_costQo�;uY�+       ��K	bY���A�*

logging/current_cost^n�;A�}+       ��K	����A�*

logging/current_cost�l�;���&+       ��K	�����A�*

logging/current_cost�l�;,���+       ��K	B遨�A�*

logging/current_cost�j�;�(YV+       ��K	����A�*

logging/current_cost�i�;cj�+       ��K	{P���A�*

logging/current_costQi�;�Z/+       ��K	�~���A�*

logging/current_cost<h�;����+       ��K	­���A�*

logging/current_costg�;Iʔ�+       ��K	9߂��A�*

logging/current_costqf�;.=0�+       ��K	����A�*

logging/current_cost}e�;9��+       ��K	�B���A�*

logging/current_cost�d�;^��r+       ��K	Wp���A�*

logging/current_cost?d�;�_�+       ��K	͞���A�*

logging/current_costc�;&��+       ��K	�҃��A�*

logging/current_cost�b�;�Q�+       ��K	����A�*

logging/current_cost�a�;o]�+       ��K	3���A�*

logging/current_cost�a�;��V�+       ��K	Ff���A�*

logging/current_cost�`�;d�U+       ��K	����A�*

logging/current_costs`�;� zG+       ��K	�Ʉ��A�*

logging/current_cost�_�;|��+       ��K	=����A�*

logging/current_costz_�;j�D+       ��K	D'���A�*

logging/current_cost^�;=��+       ��K	����A�*

logging/current_cost$^�;,4�I+       ��K	BÅ��A�*

logging/current_cost�]�;�i�+       ��K	���A�*

logging/current_cost]�;�(`~+       ��K	F;���A�*

logging/current_cost�\�;���u+       ��K	u���A�*

logging/current_coste\�;��+       ��K	+����A�*

logging/current_cost�[�;uZqn+       ��K	+����A�*

logging/current_cost�[�;�� +       ��K	`(���A�*

logging/current_costu[�;A0�+       ��K	]���A�*

logging/current_cost[�;{a�+       ��K	�����A�*

logging/current_cost�Z�;��+       ��K	 ʇ��A�*

logging/current_cost/Z�;h?�++       ��K	�����A�*

logging/current_cost�U�;��+       ��K	M-���A�*

logging/current_cost�D�;��n+       ��K	J_���A�*

logging/current_cost�8�;�wl+       ��K	k����A�*

logging/current_cost�0�;��A�+       ��K	r҈��A�*

logging/current_cost],�;�� +       ��K	�	���A�*

logging/current_cost,�;�b�e+       ��K	n9���A�*

logging/current_costK*�;�?+       ��K	{n���A�*

logging/current_cost�(�;�<+       ��K	g����A�*

logging/current_cost('�;�N&�+       ��K	�҉��A�*

logging/current_cost$�;���+       ��K	����A�*

logging/current_costT"�;V�`+       ��K	�0���A�*

logging/current_cost� �;N3�B+       ��K		`���A�*

logging/current_cost �;tFH+       ��K	؏���A�*

logging/current_cost��;�X�+       ��K	�����A�*

logging/current_cost\�;���+       ��K	1슨�A�*

logging/current_cost��;%�k+       ��K	����A�*

logging/current_cost�;M��+       ��K	�N���A�*

logging/current_cost��;ӀRT+       ��K	����A�*

logging/current_cost��;���+       ��K	����A�*

logging/current_cost��;�FB	+       ��K	���A�*

logging/current_cost��;G��6+       ��K	�+���A�*

logging/current_costT�;M+       ��K	\���A�*

logging/current_cost��;}V�U+       ��K	����A�*

logging/current_costq�;���+       ��K	�����A�*

logging/current_cost�;����+       ��K	Q패�A�*

logging/current_costk�;�<|q+       ��K	����A�*

logging/current_cost��;�s �+       ��K	[L���A�*

logging/current_cost�;��=�+       ��K	�|���A�*

logging/current_cost��;�L��+       ��K	����A�*

logging/current_costR�;YvA+       ��K	a܍��A�*

logging/current_cost��;����+       ��K	����A�*

logging/current_costU�;i��K+       ��K	S;���A�*

logging/current_cost��;J.I�+       ��K	�m���A�*

logging/current_cost��;��Q+       ��K	����A�*

logging/current_cost6�;�]�+       ��K	�Ɏ��A�*

logging/current_cost��;��2+       ��K	�����A�*

logging/current_cost��;���+       ��K	�)���A�*

logging/current_cost��;�P0++       ��K	�X���A�*

logging/current_cost��;��G�+       ��K	f����A�*

logging/current_cost	�;��ھ+       ��K	]����A�*

logging/current_cost��;���p+       ��K	�叨�A�*

logging/current_cost��;��KP+       ��K	����A�*

logging/current_cost��;��{�+       ��K	�A���A�*

logging/current_cost��;0C��+       ��K	^o���A�*

logging/current_costX�;E��$+       ��K	˟���A�*

logging/current_cost��;��m+       ��K	�ΐ��A�*

logging/current_cost��;�X��+       ��K	L����A�*

logging/current_cost��;��.�+       ��K	�A���A�*

logging/current_cost��;�z6M+       ��K	�s���A�*

logging/current_cost0�;w��E+       ��K	�����A�*

logging/current_cost�;��-+       ��K	�֑��A�*

logging/current_cost��;GV��+       ��K	/���A�*

logging/current_cost��;Ծ3p+       ��K	SA���A�*

logging/current_cost�;J#�+       ��K	t���A�*

logging/current_cost(�;�7+       ��K	l����A�*

logging/current_cost+�;&W�+       ��K	�Ԓ��A�*

logging/current_cost�;�(��+       ��K	����A�*

logging/current_cost��;�'�+       ��K	�2���A�*

logging/current_cost��;�J�+       ��K	ba���A�*

logging/current_cost9�;��آ+       ��K	挓��A�*

logging/current_cost��;���?+       ��K	Z��A�*

logging/current_cost�;���+       ��K	#��A�*

logging/current_cost��;9E��+       ��K	i���A�*

logging/current_cost��;<~�5+       ��K	�K���A�*

logging/current_costY�;����+       ��K	$x���A�*

logging/current_cost��;k^�u+       ��K	>����A�*

logging/current_cost��;N��z+       ��K	3ה��A�*

logging/current_costE�;�gh+       ��K	����A�*

logging/current_cost��;�bl+       ��K		3���A�*

logging/current_cost��;k��+       ��K	e���A�*

logging/current_costG�;(�L~+       ��K	�����A�*

logging/current_cost��;��a)+       ��K	�����A�*

logging/current_costA�;�5>�+       ��K	v����A�*

logging/current_cost��;wU��+       ��K	;$���A�*

logging/current_cost{�;s+       ��K	KQ���A�*

logging/current_cost��;����+       ��K	�����A�*

logging/current_costD�;�ڪ�+       ��K	5����A�*

logging/current_cost`�;u^j*+       ��K	ݖ��A�*

logging/current_cost��;5h��+       ��K	����A�*

logging/current_cost��;���+       ��K	�;���A�*

logging/current_costP�;�A�+       ��K	(h���A�*

logging/current_costj�;��Wd+       ��K	�����A�*

logging/current_cost�;(g�o+       ��K	#×��A�*

logging/current_cost�;�&�+       ��K	��A�*

logging/current_cost��;�O��+       ��K	$���A�*

logging/current_costU�;̝�_+       ��K	�X���A�*

logging/current_cost>�;Q&05+       ��K	Ί���A�*

logging/current_cost�;�4�+       ��K	�����A�*

logging/current_cost��;`���+       ��K	�瘨�A�*

logging/current_cost��;��t�+       ��K	����A�*

logging/current_cost��;�'g+       ��K	%G���A�*

logging/current_costD�;��09+       ��K	�t���A�*

logging/current_cost@�;�;�B+       ��K	�����A�*

logging/current_cost��;L�2+       ��K	.֙��A�*

logging/current_cost#�;�둔+       ��K	����A�*

logging/current_cost��;ޘ+       ��K	�4���A�*

logging/current_cost �;�]G�+       ��K	�c���A�*

logging/current_cost��;W�U�+       ��K	�����A�*

logging/current_cost��;F���+       ��K	~Ț��A�*

logging/current_cost�;��Y+       ��K	�����A�*

logging/current_cost��;hW�+       ��K	]%���A�*

logging/current_costL�;�w�+       ��K	�V���A�*

logging/current_cost*�;���c+       ��K	�����A�*

logging/current_cost��;^�zf+       ��K	U����A�*

logging/current_costr�;�Ǆ�+       ��K	�ᛨ�A�*

logging/current_cost��;��ӥ+       ��K	����A�*

logging/current_costV�;�1�+       ��K	�C���A�*

logging/current_cost��;�ɫd+       ��K	?s���A�*

logging/current_cost��;p�[+       ��K	R����A�*

logging/current_cost�;AE+       ��K	�؜��A�*

logging/current_cost�;�`�+       ��K	$���A�*

logging/current_cost��;7�gk+       ��K	�6���A�*

logging/current_costD�;�َ�+       ��K	�f���A�*

logging/current_cost��;��b�+       ��K	�����A�*

logging/current_cost��;{��+       ��K	���A�*

logging/current_costa�;ɗ�+       ��K	���A�*

logging/current_cost��;]^O+       ��K	n���A�*

logging/current_cost��;[��+       ��K	�M���A�*

logging/current_cost#�;�̼+       ��K	�|���A�*

logging/current_costu�;�{�+       ��K	�����A�*

logging/current_cost��;QKP/+       ��K	ޞ��A�*

logging/current_cost��;D��+       ��K	����A�*

logging/current_cost��;*x��+       ��K	�<���A�*

logging/current_cost�;>�m+       ��K	m���A�*

logging/current_cost�;)�#-+       ��K	Z����A�*

logging/current_cost��;�7XC+       ��K	�ɟ��A�*

logging/current_costM�;�JH�+       ��K	�����A�*

logging/current_cost��;��dX+       ��K	�(���A�*

logging/current_cost��;E���+       ��K	�W���A�*

logging/current_cost�;��b+       ��K	����A�*

logging/current_cost��;w���+       ��K	�����A�*

logging/current_costq�;��D�+       ��K	�ߠ��A�*

logging/current_cost��;�l��+       ��K	����A�*

logging/current_cost|�;�b��+       ��K	�7���A�*

logging/current_cost�;�Tgj+       ��K	*f���A�*

logging/current_costD�;~��
+       ��K	�����A�*

logging/current_cost[�;�z�+       ��K	�����A�*

logging/current_cost+�;"�+       ��K	�롨�A�*

logging/current_cost��;����+       ��K	���A�*

logging/current_cost��;��?j+       ��K	eH���A�*

logging/current_costz�;��N+       ��K	�t���A�*

logging/current_cost��;}�t�+       ��K	ơ���A�*

logging/current_cost��;Mo �+       ��K	�Ϣ��A�*

logging/current_cost�;�c�+       ��K	����A�*

logging/current_costq�;o�}�+       ��K	5+���A�*

logging/current_cost��;k<�2+       ��K	�Z���A�*

logging/current_cost��;�+       ��K	�����A�*

logging/current_cost��;��+       ��K	�����A�*

logging/current_cost&�;Z��+       ��K	壨�A�*

logging/current_cost��;=�6�+       ��K	y���A�*

logging/current_costU�;�p&1+       ��K	�@���A�*

logging/current_cost��;^�+       ��K	�m���A�*

logging/current_costP�;\�#j+       ��K	�����A�*

logging/current_cost^�;�̽�+       ��K	kǤ��A�*

logging/current_costY�;�ٜ}+       ��K	�����A�*

logging/current_cost�;�g\+       ��K	"���A� *

logging/current_cost��;�5+       ��K	TW���A� *

logging/current_cost��;[d�8+       ��K	y����A� *

logging/current_cost��;�'�"+       ��K	ֽ���A� *

logging/current_cost��;^a(+       ��K	)����A� *

logging/current_costk�;�D%+       ��K	����A� *

logging/current_cost��;l��+       ��K	P���A� *

logging/current_cost��;���+       ��K	����A� *

logging/current_costd�;}���+       ��K	쮦��A� *

logging/current_cost�;����+       ��K	lন�A� *

logging/current_cost��;|�!+       ��K	b���A� *

logging/current_cost�;9l)�+       ��K	hD���A� *

logging/current_cost�;��ĩ+       ��K	Is���A� *

logging/current_costi�;9�-+       ��K	﷧��A� *

logging/current_cost��;�k�h+       ��K	9駨�A� *

logging/current_cost��;���+       ��K	R���A� *

logging/current_costn�;Y�|+       ��K	�H���A� *

logging/current_cost+�;�9 +       ��K	y���A� *

logging/current_cost��;����+       ��K	`����A� *

logging/current_cost��;��݃+       ��K	'ը��A� *

logging/current_costa�;7�(+       ��K	����A� *

logging/current_costf�;�ӊ+       ��K	�4���A� *

logging/current_cost��;�B��+       ��K	�b���A� *

logging/current_cost��;�.#@+       ��K	E����A� *

logging/current_costU�;�V+       ��K	h����A� *

logging/current_cost0�;����+       ��K	�難�A�!*

logging/current_cost��;��.�+       ��K	6���A�!*

logging/current_costI�;��*+       ��K	nM���A�!*

logging/current_costZ�;�F#+       ��K	]|���A�!*

logging/current_cost��;Xx�+       ��K	�����A�!*

logging/current_cost9�;�3+       ��K	�ڪ��A�!*

logging/current_cost��;,Ub�+       ��K	�	���A�!*

logging/current_cost=�;.���+       ��K	�:���A�!*

logging/current_cost�;O:�+       ��K	�i���A�!*

logging/current_costO�;�:Ȕ+       ��K	�����A�!*

logging/current_cost �;=+��+       ��K	�ƫ��A�!*

logging/current_cost	�;�y��+       ��K	)����A�!*

logging/current_costf�;��W]+       ��K	�"���A�!*

logging/current_cost�;5[+       ��K	bQ���A�!*

logging/current_cost7�;o-�+       ��K	�}���A�!*

logging/current_costE�;Ε�+       ��K	r����A�!*

logging/current_cost�;�+       ��K	�ڬ��A�!*

logging/current_costM�;��m�+       ��K	���A�!*

logging/current_cost�;~}�[+       ��K	�4���A�!*

logging/current_costf�;��\+       ��K	b���A�!*

logging/current_cost=�;��+       ��K	�����A�!*

logging/current_cost��;���+       ��K	����A�!*

logging/current_cost,�;ϓx+       ��K	r뭨�A�!*

logging/current_cost��;dE�+       ��K	v���A�!*

logging/current_cost$�;�0��+       ��K	�E���A�!*

logging/current_cost]�;M��+       ��K	
t���A�!*

logging/current_cost(�;Gl+       ��K	򢮨�A�"*

logging/current_cost��;b���+       ��K	8Ϯ��A�"*

logging/current_cost��;v��{+       ��K	����A�"*

logging/current_cost�;�j�$+       ��K	�*���A�"*

logging/current_cost��;�Ԇz+       ��K	�W���A�"*

logging/current_cost[�;�G�e+       ��K	�����A�"*

logging/current_costS�;�W"+       ��K	$����A�"*

logging/current_costy�;pר�+       ��K	�㯨�A�"*

logging/current_cost��;ý�+       ��K	'���A�"*

logging/current_costn�;fI0�+       ��K	�@���A�"*

logging/current_cost �;�%+       ��K	�m���A�"*

logging/current_costp�;ʧ_+       ��K	�����A�"*

logging/current_cost��;�N��+       ��K	�̰��A�"*

logging/current_cost'�;��s+       ��K	�����A�"*

logging/current_cost��;M�K�+       ��K	�+���A�"*

logging/current_cost��;�	��+       ��K	Z���A�"*

logging/current_costB�;�� �+       ��K	l����A�"*

logging/current_cost��;�/�+       ��K	�����A�"*

logging/current_cost��;���+       ��K	�㱨�A�"*

logging/current_cost��;X���+       ��K	����A�"*

logging/current_cost��;8e�"+       ��K	>���A�"*

logging/current_cost,�;��6�+       ��K	s���A�"*

logging/current_cost��;�q�+       ��K	�����A�"*

logging/current_cost+�;t�n�+       ��K	[Բ��A�"*

logging/current_cost�;f�^+       ��K	����A�"*

logging/current_cost��;Ud�M+       ��K	�,���A�#*

logging/current_costg�;�"s+       ��K	�`���A�#*

logging/current_cost��;	ۋ+       ��K	Ќ���A�#*

logging/current_cost�;K�~�+       ��K	�����A�#*

logging/current_cost��;	t�+       ��K	����A�#*

logging/current_cost��;��t+       ��K	` ���A�#*

logging/current_costG�;=]��+       ��K	0M���A�#*

logging/current_cost��;�^o�+       ��K	}{���A�#*

logging/current_cost��;Z���+       ��K	�����A�#*

logging/current_cost��;��Q�+       ��K	ڴ��A�#*

logging/current_cost��;+��+       ��K	����A�#*

logging/current_costQ�;,��+       ��K	�7���A�#*

logging/current_cost��;H�J+       ��K	'i���A�#*

logging/current_cost�;��+       ��K	����A�#*

logging/current_cost��;�[ߊ+       ��K	�ǵ��A�#*

logging/current_cost�;6~4+       ��K	�����A�#*

logging/current_costO�;�y�+       ��K	)(���A�#*

logging/current_costj�;Ǐ+       ��K	�V���A�#*

logging/current_cost)�;��RK+       ��K	ㆶ��A�#*

logging/current_cost��;����+       ��K	[����A�#*

logging/current_cost��;!�l�+       ��K	~㶨�A�#*

logging/current_cost�;</@+       ��K	����A�#*

logging/current_cost��;��U�+       ��K	'C���A�#*

logging/current_costG�;��H+       ��K	ks���A�#*

logging/current_cost��;�.m+       ��K	�����A�#*

logging/current_cost��;���p+       ��K	�з��A�#*

logging/current_cost��;��v+       ��K	����A�$*

logging/current_costN�;R���+       ��K	�2���A�$*

logging/current_costL�;R�}�+       ��K	�a���A�$*

logging/current_cost��;��H�+       ��K	����A�$*

logging/current_cost\�;5 Ve+       ��K	5����A�$*

logging/current_cost��;Ys�+       ��K	�����A�$*

logging/current_cost��;�8^�+       ��K	&���A�$*

logging/current_cost)�;���+       ��K	MK���A�$*

logging/current_cost��;�~+       ��K	�x���A�$*

logging/current_cost5�;���*+       ��K	ӧ���A�$*

logging/current_costG�;��k�+       ��K	eԹ��A�$*

logging/current_cost��;߫y+       ��K	D���A�$*

logging/current_cost��;����+       ��K	�.���A�$*

logging/current_cost��;�/F+       ��K	3]���A�$*

logging/current_cost��;_��+       ��K	�����A�$*

logging/current_cost��;1)V�+       ��K	8����A�$*

logging/current_costU�;R���+       ��K	�庨�A�$*

logging/current_cost��;���+       ��K	A���A�$*

logging/current_cost��;����+       ��K	�A���A�$*

logging/current_costT�;��:�+       ��K	u���A�$*

logging/current_cost��;��&-+       ��K	�λ��A�$*

logging/current_cost��;mm�O+       ��K	����A�$*

logging/current_cost�;�ph7+       ��K	9���A�$*

logging/current_costg�;K�� +       ��K	-z���A�$*

logging/current_cost��;�^i;+       ��K	����A�$*

logging/current_cost`�;��+       ��K	�輨�A�$*

logging/current_cost��;�� +       ��K	"���A�%*

logging/current_costc�;�O:�+       ��K	ZW���A�%*

logging/current_cost��;��SS+       ��K	퉽��A�%*

logging/current_cost��;a�(0+       ��K	b����A�%*

logging/current_cost&�;��c+       ��K	�����A�%*

logging/current_cost��;?��+       ��K	m/���A�%*

logging/current_cost,�;>W�+       ��K	�b���A�%*

logging/current_cost��;���+       ��K	璾��A�%*

logging/current_cost��;��:+       ��K	Nƾ��A�%*

logging/current_costV�;A�g�+       ��K	�����A�%*

logging/current_cost��;�[P�+       ��K	%���A�%*

logging/current_cost��;�׬W+       ��K	3W���A�%*

logging/current_cost-�;mA]+       ��K	V����A�%*

logging/current_costi�;F�R+       ��K	����A�%*

logging/current_cost��;j�[+       ��K	�鿨�A�%*

logging/current_cost��;�×�+       ��K	�!���A�%*

logging/current_cost`�;{���+       ��K	�R���A�%*

logging/current_cost �;��+       ��K	r����A�%*

logging/current_cost��;ÿ�{+       ��K	�����A�%*

logging/current_cost��;S�W+       ��K	v����A�%*

logging/current_costT�;q3+       ��K	'&���A�%*

logging/current_cost��;�C
+       ��K	�S���A�%*

logging/current_cost��;ѸXU+       ��K	����A�%*

logging/current_cost��;�A��+       ��K	g����A�%*

logging/current_cost��;~��=+       ��K	�����A�%*

logging/current_costz�;���+       ��K	.¨�A�&*

logging/current_cost|�;��Z�+       ��K	iK¨�A�&*

logging/current_cost&�;�%�n+       ��K	�y¨�A�&*

logging/current_cost#�;܊"�+       ��K	z�¨�A�&*

logging/current_costo�;�/+       ��K	��¨�A�&*

logging/current_cost��;�(�@+       ��K	=è�A�&*

logging/current_costn�;��5+       ��K	94è�A�&*

logging/current_cost��;��<�+       ��K	�aè�A�&*

logging/current_cost�;��=�+       ��K	�è�A�&*

logging/current_cost��;V/wF+       ��K	N�è�A�&*

logging/current_costM�;VA+       ��K	��è�A�&*

logging/current_cost��;�64+       ��K	�%Ĩ�A�&*

logging/current_cost�;�{��+       ��K	�UĨ�A�&*

logging/current_cost��;�}+       ��K	��Ĩ�A�&*

logging/current_costa�;��v�+       ��K	A�Ĩ�A�&*

logging/current_costC�;uE`"+       ��K	��Ĩ�A�&*

logging/current_cost��;�G�;+       ��K	SŨ�A�&*

logging/current_cost��;�u+       ��K	&?Ũ�A�&*

logging/current_cost��;a�+       ��K	�oŨ�A�&*

logging/current_cost�;�<-+       ��K	_�Ũ�A�&*

logging/current_cost��;,�C�+       ��K	��Ũ�A�&*

logging/current_cost��;9 �3+       ��K	T�Ũ�A�&*

logging/current_cost��;^FLl+       ��K	Y,ƨ�A�&*

logging/current_cost��;�i��+       ��K	�Yƨ�A�&*

logging/current_cost��;yǿ?+       ��K	��ƨ�A�&*

logging/current_cost��;5�§+       ��K	��ƨ�A�&*

logging/current_costA�;\��+       ��K	��ƨ�A�'*

logging/current_cost��;a+[&+       ��K	�Ǩ�A�'*

logging/current_cost|�;�j�i+       ��K	UCǨ�A�'*

logging/current_cost��;�"��+       ��K	�oǨ�A�'*

logging/current_cost��;M�nN+       ��K	��Ǩ�A�'*

logging/current_cost��;s�tS+       ��K	��Ǩ�A�'*

logging/current_cost��;����+       ��K	��Ǩ�A�'*

logging/current_cost8�;B��+       ��K	�'Ȩ�A�'*

logging/current_cost�;s��q+       ��K	/UȨ�A�'*

logging/current_costI�;W3.?+       ��K	ЊȨ�A�'*

logging/current_cost��;Ҝ�r+       ��K	��Ȩ�A�'*

logging/current_costV�;�xw+       ��K	t�Ȩ�A�'*

logging/current_cost��;���+       ��K	�ɨ�A�'*

logging/current_costG�;h�0+       ��K	!Eɨ�A�'*

logging/current_cost��;]m��+       ��K	�sɨ�A�'*

logging/current_cost�;HWܕ+       ��K	e�ɨ�A�'*

logging/current_costh�;�M��+       ��K	�ɨ�A�'*

logging/current_cost��;G��O+       ��K	M�ɨ�A�'*

logging/current_cost
�;�i�g+       ��K	�0ʨ�A�'*

logging/current_cost�;D��+       ��K	7^ʨ�A�'*

logging/current_cost��;�_��+       ��K	ǌʨ�A�'*

logging/current_cost'�;�y�+       ��K	ϻʨ�A�'*

logging/current_cost��;�[]+       ��K	�ʨ�A�'*

logging/current_cost@�;���+       ��K	{˨�A�'*

logging/current_costA�;�{5+       ��K	�J˨�A�'*

logging/current_cost^�;�;+       ��K	�x˨�A�(*

logging/current_cost��;W�_�+       ��K	l�˨�A�(*

logging/current_cost��;`�[+       ��K	��˨�A�(*

logging/current_cost��;M7Ѝ+       ��K	�̨�A�(*

logging/current_cost��;R�K�+       ��K	�4̨�A�(*

logging/current_cost��;2�_m+       ��K	�d̨�A�(*

logging/current_cost^�;p�IH+       ��K	n�̨�A�(*

logging/current_cost��;>��++       ��K	��̨�A�(*

logging/current_costj�;�1Y+       ��K	��̨�A�(*

logging/current_costR�;qn��+       ��K	�!ͨ�A�(*

logging/current_cost'�;�۾+       ��K	kOͨ�A�(*

logging/current_cost��;|�<7+       ��K	}ͨ�A�(*

logging/current_costx�;_�w�+       ��K	 �ͨ�A�(*

logging/current_cost~�; 5�Z+       ��K	$�ͨ�A�(*

logging/current_costq�;�^��+       ��K	�Ψ�A�(*

logging/current_cost��; J��+       ��K	5Ψ�A�(*

logging/current_cost��;�+       ��K	.fΨ�A�(*

logging/current_cost��;�b�+       ��K	֔Ψ�A�(*

logging/current_cost��;��`r+       ��K	��Ψ�A�(*

logging/current_cost	�;���r+       ��K	��Ψ�A�(*

logging/current_cost!�;L5�+       ��K	�$Ϩ�A�(*

logging/current_costz�;Qq�]+       ��K	�SϨ�A�(*

logging/current_cost��;���D+       ��K	-�Ϩ�A�(*

logging/current_cost��;ټ��+       ��K	)�Ϩ�A�(*

logging/current_cost�;�6�r+       ��K	��Ϩ�A�(*

logging/current_costm�;�0�+       ��K	�Ш�A�(*

logging/current_cost��;�+^+       ��K	;Ш�A�)*

logging/current_costZ�;�td+       ��K	�iШ�A�)*

logging/current_cost��;e�`�+       ��K	�Ш�A�)*

logging/current_cost��;2�E�+       ��K	��Ш�A�)*

logging/current_cost��;���:+       ��K	�Ш�A�)*

logging/current_costo�;a"?P+       ��K	�)Ѩ�A�)*

logging/current_costY�;�a��+       ��K	�^Ѩ�A�)*

logging/current_cost[�;�5�+       ��K	�Ѩ�A�)*

logging/current_cost��;geq+       ��K	��Ѩ�A�)*

logging/current_cost��;F��a+       ��K	%�Ѩ�A�)*

logging/current_cost��;�\F<+       ��K	9Ҩ�A�)*

logging/current_cost{�;���&+       ��K	�Ҩ�A�)*

logging/current_cost��;F�#+       ��K	x�Ҩ�A�)*

logging/current_cost��;FuN�+       ��K	�(Ө�A�)*

logging/current_cost��;\�LR+       ��K	4`Ө�A�)*

logging/current_cost��;�'�0+       ��K	��Ө�A�)*

logging/current_cost�;gw"�+       ��K	T�Ө�A�)*

logging/current_cost��;fZ�l+       ��K	�"Ԩ�A�)*

logging/current_costg�;�Wû+       ��K	GWԨ�A�)*

logging/current_cost��;�}k+       ��K	Z�Ԩ�A�)*

logging/current_cost�;3���+       ��K	��Ԩ�A�)*

logging/current_cost��;L}�+       ��K	��Ԩ�A�)*

logging/current_cost��;�,�O+       ��K	5#ը�A�)*

logging/current_costN�;vwm�+       ��K	`ը�A�)*

logging/current_cost��;�i��+       ��K	d�ը�A�)*

logging/current_costW�;~7�`+       ��K	Y�ը�A�)*

logging/current_cost��;���+       ��K	'֨�A�**

logging/current_cost��;��b�+       ��K	�>֨�A�**

logging/current_cost9�;�Ùo+       ��K	�s֨�A�**

logging/current_cost��;�g��+       ��K	F�֨�A�**

logging/current_cost2�;�~={+       ��K	2�֨�A�**

logging/current_cost��;>_?+       ��K	!ר�A�**

logging/current_costb�;	��+       ��K	o0ר�A�**

logging/current_cost�;��&�+       ��K	�aר�A�**

logging/current_cost�;���+       ��K	l�ר�A�**

logging/current_cost]�;�h�+       ��K	��ר�A�**

logging/current_cost��;`N+       ��K	I�ר�A�**

logging/current_cost��;��Rc+       ��K	2#ب�A�**

logging/current_cost�;P]�+       ��K	�Uب�A�**

logging/current_cost��;Yr�+       ��K	��ب�A�**

logging/current_cost�;�z�X+       ��K	<�ب�A�**

logging/current_costw�;l]F�+       ��K	O�ب�A�**

logging/current_cost^�;F~I$+       ��K	�2٨�A�**

logging/current_cost��;JO
2+       ��K	�e٨�A�**

logging/current_cost��;��5+       ��K	�٨�A�**

logging/current_cost��;?��c+       ��K	��٨�A�**

logging/current_cost��;1�J�+       ��K	��٨�A�**

logging/current_cost0�;��+       ��K	�!ڨ�A�**

logging/current_cost��;���+       ��K	�Vڨ�A�**

logging/current_cost�;Gܿ�+       ��K	��ڨ�A�**

logging/current_costI�;�L�+       ��K	f�ڨ�A�**

logging/current_cost��;�ِ�+       ��K	��ڨ�A�+*

logging/current_cost��;���+       ��K	�ۨ�A�+*

logging/current_costM�;��+       ��K	�Hۨ�A�+*

logging/current_cost��;t���+       ��K	�{ۨ�A�+*

logging/current_cost��;��_�+       ��K	��ۨ�A�+*

logging/current_cost��;5b:+       ��K	C�ۨ�A�+*

logging/current_cost�;L�M+       ��K	,ܨ�A�+*

logging/current_cost��;�e��+       ��K	hfܨ�A�+*

logging/current_cost��;���+       ��K	�ܨ�A�+*

logging/current_cost��;�u�&+       ��K	��ܨ�A�+*

logging/current_cost��;��z+       ��K	�ݨ�A�+*

logging/current_cost�;N�Z�+       ��K	�Aݨ�A�+*

logging/current_cost��;wb�+       ��K	�ݨ�A�+*

logging/current_cost��;�Gu�+       ��K	��ݨ�A�+*

logging/current_costq�;M���+       ��K	?�ݨ�A�+*

logging/current_costE�;�:q�+       ��K	 ި�A�+*

logging/current_cost�;��{�+       ��K	�Sި�A�+*

logging/current_cost��;�sw5+       ��K	јި�A�+*

logging/current_cost#�;���+       ��K	X�ި�A�+*

logging/current_cost�;<E��+       ��K	i4ߨ�A�+*

logging/current_costV�;W+9,+       ��K	)dߨ�A�+*

logging/current_cost��;|Mh+       ��K	{�ߨ�A�+*

logging/current_cost��;��D�+       ��K	�ߨ�A�+*

logging/current_cost�;E�å+       ��K	Q��A�+*

logging/current_cost��;a�F<+       ��K	�@��A�+*

logging/current_cost"�;���+       ��K	�q��A�+*

logging/current_costm�;P��i+       ��K	\���A�,*

logging/current_cost��;�Q��+       ��K	e���A�,*

logging/current_cost��;���*+       ��K	(��A�,*

logging/current_cost^�;h��+       ��K	Z��A�,*

logging/current_cost0�;��G+       ��K	c���A�,*

logging/current_costF�;�[;+       ��K	����A�,*

logging/current_costC�;s�q+       ��K	I��A�,*

logging/current_cost��;�g��+       ��K	$C��A�,*

logging/current_cost�;�'��+       ��K	�u��A�,*

logging/current_cost��;�Z8+       ��K	���A�,*

logging/current_cost{�;����+       ��K	���A�,*

logging/current_cost��;��+       ��K	�&��A�,*

logging/current_cost��;��j_+       ��K	o^��A�,*

logging/current_cost��;X�
+       ��K	1���A�,*

logging/current_cost��;i �+       ��K	����A�,*

logging/current_cost1�;B#�h+       ��K	
��A�,*

logging/current_cost��;��b+       ��K	=R��A�,*

logging/current_cost��;�q��+       ��K	����A�,*

logging/current_cost�;�IR+       ��K	3���A�,*

logging/current_cost��;tH�+       ��K	����A�,*

logging/current_cost��;߿c+       ��K	��A�,*

logging/current_cost,�;���8+       ��K	�Q��A�,*

logging/current_costf�;3Y�*+       ��K	���A�,*

logging/current_cost0�;���+       ��K	i���A�,*

logging/current_cost,�;�/��+       ��K	����A�,*

logging/current_cost��;�4�+       ��K	���A�-*

logging/current_cost��;KK�+       ��K	_P��A�-*

logging/current_cost��;��!m+       ��K	*���A�-*

logging/current_cost��;�ѵ+       ��K	S���A�-*

logging/current_cost�;�y�+       ��K	����A�-*

logging/current_cost �;!P��+       ��K	<��A�-*

logging/current_cost��;[d�+       ��K	D{��A�-*

logging/current_cost>�;&qC�+       ��K	���A�-*

logging/current_costJ�;ʁ��+       ��K	3���A�-*

logging/current_costu�;����+       ��K	t��A�-*

logging/current_cost6�;b}F�+       ��K	]U��A�-*

logging/current_cost�;c�{+       ��K	l���A�-*

logging/current_cost��;��A+       ��K	����A�-*

logging/current_cost��;/��+       ��K	����A�-*

logging/current_cost9�;E���+       ��K	� ��A�-*

logging/current_cost��;���+       ��K	V��A�-*

logging/current_cost�;��Q+       ��K	Q���A�-*

logging/current_cost�;�z'�+       ��K	(���A�-*

logging/current_cost��;͇�+       ��K	����A�-*

logging/current_cost��;q�W+       ��K	�#��A�-*

logging/current_costI�;�K��+       ��K	+Y��A�-*

logging/current_cost��;�YR�+       ��K	8���A�-*

logging/current_costP�;l2{�+       ��K	����A�-*

logging/current_cost��;��G�+       ��K	�!��A�-*

logging/current_costv�;j��+       ��K	to��A�-*

logging/current_cost��;��+       ��K	3���A�-*

logging/current_cost8�;k;��+       ��K	6���A�.*

logging/current_cost��;Bԓ+       ��K	�-��A�.*

logging/current_cost��;���+       ��K	sh��A�.*

logging/current_costA�;���+       ��K	ן��A�.*

logging/current_cost��;nP֠+       ��K	���A�.*

logging/current_costg�;�e�+       ��K	H ���A�.*

logging/current_cost��;ܷ�+       ��K	0���A�.*

logging/current_costS�;ep
+       ��K	�b���A�.*

logging/current_cost��;L{�v+       ��K	9����A�.*

logging/current_cost'�;�H�+       ��K	����A�.*

logging/current_cost��;x(�+       ��K	���A�.*

logging/current_cost��;���+       ��K	�7��A�.*

logging/current_cost5�;
���+       ��K	�j��A�.*

logging/current_cost��;L�4�+       ��K	����A�.*

logging/current_cost��;[���+       ��K	Z���A�.*

logging/current_cost��;�&�^+       ��K	���A�.*

logging/current_cost��;�(�+       ��K	A3��A�.*

logging/current_cost!�;-�E+       ��K	�b��A�.*

logging/current_costO�;e�(�+       ��K	I���A�.*

logging/current_costj�;Z�+�+       ��K	����A�.*

logging/current_cost�;SW2+       ��K	��A�.*

logging/current_cost6�;�9�+       ��K	A��A�.*

logging/current_cost$�;lw��+       ��K	+r��A�.*

logging/current_cost~�;n���+       ��K	����A�.*

logging/current_cost��;�Po�+       ��K	����A�.*

logging/current_cost��;�wU+       ��K	u
��A�.*

logging/current_cost�;�R�+       ��K	59��A�/*

logging/current_cost	�;05�N+       ��K	�h��A�/*

logging/current_cost��;�#ߠ+       ��K	m���A�/*

logging/current_cost_�;�zթ+       ��K	+���A�/*

logging/current_cost��;���v+       ��K	C���A�/*

logging/current_cost��;�$T�+       ��K	$��A�/*

logging/current_cost��;x.�+       ��K	�S��A�/*

logging/current_cost��;�g�e+       ��K	=���A�/*

logging/current_costE�;q��+       ��K	)���A�/*

logging/current_cost��;��1�+       ��K	����A�/*

logging/current_costQ�;���m+       ��K	s ��A�/*

logging/current_cost��;,j�+       ��K	PO��A�/*

logging/current_cost��;�ͷ+       ��K	�{��A�/*

logging/current_cost�;8{�+       ��K	���A�/*

logging/current_cost��;�d��+       ��K	����A�/*

logging/current_cost��;��_+       ��K	:���A�/*

logging/current_costc�;c,�+       ��K	~@���A�/*

logging/current_costY�;����+       ��K	�p���A�/*

logging/current_cost��;%s+       ��K	{����A�/*

logging/current_cost/�;Td�+       ��K	�����A�/*

logging/current_costl�;�H�+       ��K	s���A�/*

logging/current_cost��;2Җ+       ��K	_L���A�/*

logging/current_cost��;�M�+       ��K	�����A�/*

logging/current_cost��;�c[>+       ��K	�����A�/*

logging/current_cost0�;XCR+       ��K	�����A�/*

logging/current_cost��;C��D+       ��K	�:���A�0*

logging/current_cost��;�x+       ��K	�q���A�0*

logging/current_cost��;y�R�+       ��K	����A�0*

logging/current_cost��;o`��+       ��K	o����A�0*

logging/current_cost=�;�c�+       ��K	����A�0*

logging/current_cost8�;�٨+       ��K	�<���A�0*

logging/current_cost[�;_�ù+       ��K	"s���A�0*

logging/current_cost}�;�튉+       ��K	�����A�0*

logging/current_cost��;+���+       ��K	�����A�0*

logging/current_cost��;��+       ��K	�X���A�0*

logging/current_cost��;X��+       ��K	�����A�0*

logging/current_cost��;�Q�+       ��K	m����A�0*

logging/current_cost��;F��+       ��K	n���A�0*

logging/current_cost�;����+       ��K	cC���A�0*

logging/current_costX�;�6�+       ��K	z���A�0*

logging/current_cost��;v��+       ��K	����A�0*

logging/current_cost��;�Qj+       ��K	$���A�0*

logging/current_cost&�;=��+       ��K	28���A�0*

logging/current_cost��;��Z[+       ��K	
g���A�0*

logging/current_cost��;��+       ��K	5����A�0*

logging/current_cost=�;bӓ+       ��K	$����A�0*

logging/current_cost&�;�G#d+       ��K		����A�0*

logging/current_costi�;g�� +       ��K	�#���A�0*

logging/current_cost��;���+       ��K	�S���A�0*

logging/current_cost1�;�Y�+       ��K	Ԡ���A�0*

logging/current_cost��;�}[+       ��K	���A�0*

logging/current_cost8�;0#�y+       ��K	�V���A�1*

logging/current_costZ�;O�Ů+       ��K	9����A�1*

logging/current_cost,�;,r^+       ��K	�����A�1*

logging/current_cost��;Kp�m+       ��K	�����A�1*

logging/current_costr�;��v+       ��K	K1���A�1*

logging/current_cost��;�s�+       ��K	�d���A�1*

logging/current_costz�;C{�+       ��K	�����A�1*

logging/current_costD�;K��+       ��K	�����A�1*

logging/current_cost��;f*��+       ��K	����A�1*

logging/current_cost3�;�65+       ��K	�0���A�1*

logging/current_cost8�;6��c+       ��K	~b���A�1*

logging/current_cost��;u��'+       ��K	b����A�1*

logging/current_costf�;b���+       ��K	%����A�1*

logging/current_cost�;7��U+       ��K	s����A�1*

logging/current_costb�;D�SG+       ��K	$���A�1*

logging/current_cost�;�t�+       ��K	�S���A�1*

logging/current_cost��;��+`+       ��K	�����A�1*

logging/current_cost)�;�3+       ��K	�����A�1*

logging/current_cost�;_A��+       ��K	�����A�1*

logging/current_cost��;��j>+       ��K	� ��A�1*

logging/current_costS�;��~+       ��K	�> ��A�1*

logging/current_cost��;��\�+       ��K	
m ��A�1*

logging/current_costj�;���+       ��K	� ��A�1*

logging/current_cost�;�O�2+       ��K	�� ��A�1*

logging/current_cost9�;�%t�+       ��K	�� ��A�1*

logging/current_cost��;����+       ��K	�0��A�2*

logging/current_cost�;r��+       ��K	/`��A�2*

logging/current_cost:�;�T`+       ��K	����A�2*

logging/current_costQ�;;f�+       ��K	t���A�2*

logging/current_cost��;�E=�+       ��K	���A�2*

logging/current_cost��;��+       ��K	c)��A�2*

logging/current_cost�;·�{+       ��K	�[��A�2*

logging/current_cost��;'��O+       ��K	f���A�2*

logging/current_cost?�;-xR+       ��K	����A�2*

logging/current_costA�;O�g+       ��K	D���A�2*

logging/current_cost�;�z��+       ��K	���A�2*

logging/current_costL�;n	�%+       ��K	�M��A�2*

logging/current_cost��;��B3+       ��K	B}��A�2*

logging/current_cost(�;�h3+       ��K	r���A�2*

logging/current_cost<�;��c�+       ��K	)���A�2*

logging/current_cost��;�u�'+       ��K	`��A�2*

logging/current_cost��;UyA�+       ��K	�>��A�2*

logging/current_cost@�;�"��+       ��K	�l��A�2*

logging/current_costx�;�� �+       ��K	���A�2*

logging/current_costj�;���+       ��K	r���A�2*

logging/current_cost��;��)+       ��K	����A�2*

logging/current_cost�;�r2+       ��K	�*��A�2*

logging/current_cost)�;�3�+       ��K	�Y��A�2*

logging/current_costG�;|�+�+       ��K	���A�2*

logging/current_costc�;�n�@+       ��K	����A�2*

logging/current_cost��;= �+       ��K	���A�2*

logging/current_costG�;Dϩ�+       ��K	���A�3*

logging/current_cost��;���+       ��K	BD��A�3*

logging/current_cost�;�߶�+       ��K	�r��A�3*

logging/current_cost�;����+       ��K	d���A�3*

logging/current_costi�;peBI+       ��K	���A�3*

logging/current_cost��;Z`�+       ��K	<���A�3*

logging/current_costI�;��l�+       ��K	�-��A�3*

logging/current_cost��;(��+       ��K	�Z��A�3*

logging/current_costZ�;h��J+       ��K	v���A�3*

logging/current_costK�;��,+       ��K	v���A�3*

logging/current_costo�;&���+       ��K	����A�3*

logging/current_costc�;��=+       ��K	T��A�3*

logging/current_cost�;��+       ��K	A��A�3*

logging/current_cost��;�1�B+       ��K	�o��A�3*

logging/current_cost��;7M�+       ��K	ݤ��A�3*

logging/current_cost�;����+       ��K	����A�3*

logging/current_costP�;�g8r+       ��K	����A�3*

logging/current_costf�;I��+       ��K	-	��A�3*

logging/current_cost��;���+       ��K	�[	��A�3*

logging/current_cost��;�,�8+       ��K	��	��A�3*

logging/current_cost��;,��+       ��K	��	��A�3*

logging/current_costg�;!8�+       ��K	��	��A�3*

logging/current_cost��;$~L�+       ��K	 
��A�3*

logging/current_cost�;��1 +       ��K	^A
��A�3*

logging/current_cost#�;Y�D�+       ��K	�n
��A�3*

logging/current_cost��;kye.+       ��K	 �
��A�3*

logging/current_cost}�;���C+       ��K	?�
��A�4*

logging/current_cost�;|���+       ��K	.�
��A�4*

logging/current_cost��;�i�$+       ��K	�(��A�4*

logging/current_cost��;�Fd+       ��K	V��A�4*

logging/current_cost��;2�)�+       ��K	���A�4*

logging/current_cost��;,�+       ��K	l���A�4*

logging/current_cost��;.++       ��K	����A�4*

logging/current_cost"�;���9+       ��K	[7��A�4*

logging/current_cost(�;��+       ��K	�s��A�4*

logging/current_cost>�;+◙+       ��K	���A�4*

logging/current_cost��;����+       ��K	����A�4*

logging/current_cost�;�TÑ+       ��K	�$��A�4*

logging/current_cost�;*��>+       ��K	�U��A�4*

logging/current_cost��;�h�+       ��K	z���A�4*

logging/current_cost�;��u+       ��K	����A�4*

logging/current_cost��;_���+       ��K	l	��A�4*

logging/current_cost?�;��$�+       ��K	�>��A�4*

logging/current_cost��;�gi�+       ��K	#o��A�4*

logging/current_cost��;��o�+       ��K	����A�4*

logging/current_cost$�;��
J+       ��K	����A�4*

logging/current_cost4�;U��@+       ��K	��A�4*

logging/current_costa�;\�=+       ��K	�G��A�4*

logging/current_costL�;�2�+       ��K	3���A�4*

logging/current_cost��;��:4+       ��K	����A�4*

logging/current_cost��;�҂{+       ��K	����A�4*

logging/current_cost'�;���+       ��K	d��A�5*

logging/current_cost��;O'?T+       ��K	�M��A�5*

logging/current_cost��;�D��+       ��K	Rz��A�5*

logging/current_cost)�;/�+       ��K	����A�5*

logging/current_cost"�;qNQ&+       ��K	����A�5*

logging/current_costJ�;�˗W+       ��K	���A�5*

logging/current_cost��;�hr+       ��K	�?��A�5*

logging/current_cost��;���J+       ��K	3n��A�5*

logging/current_cost�;�+       ��K	����A�5*

logging/current_cost��;���+       ��K	����A�5*

logging/current_cost��;�>�+       ��K	����A�5*

logging/current_cost��;�n<�+       ��K	&5��A�5*

logging/current_cost��;�� +       ��K	�j��A�5*

logging/current_cost��;@�+       ��K	���A�5*

logging/current_costr�;�1�+       ��K	%���A�5*

logging/current_cost��;�)�_+       ��K	���A�5*

logging/current_cost��;�,��+       ��K	�6��A�5*

logging/current_cost��;P�`e+       ��K	rd��A�5*

logging/current_cost��;�~�y+       ��K	t���A�5*

logging/current_costs�;-)�+       ��K	5���A�5*

logging/current_cost��;��s�+       ��K	!���A�5*

logging/current_cost��;���+       ��K	�(��A�5*

logging/current_cost�;F(��+       ��K	+Z��A�5*

logging/current_cost��;��m�+       ��K	u���A�5*

logging/current_cost��;?5�+       ��K	���A�5*

logging/current_cost��;�; =+       ��K	O���A�5*

logging/current_cost��;B<��+       ��K	���A�6*

logging/current_costQ�;�Yi�+       ��K	�D��A�6*

logging/current_cost�;q3�+       ��K	�r��A�6*

logging/current_cost�
�;�V�+       ��K	q���A�6*

logging/current_cost.	�;��`�+       ��K		���A�6*

logging/current_cost��;�H0+       ��K	\���A�6*

logging/current_cost~�;�(�+       ��K	C)��A�6*

logging/current_costl�;���+       ��K	�U��A�6*

logging/current_cost�;��Sj+       ��K	����A�6*

logging/current_cost���;��+       ��K	����A�6*

logging/current_cost���;�F��+       ��K	����A�6*

logging/current_cost:��;&i�z+       ��K	���A�6*

logging/current_cost���;�a+       ��K	�:��A�6*

logging/current_cost�;����+       ��K	eh��A�6*

logging/current_cost�;�L�A+       ��K	����A�6*

logging/current_cost_�;Y+       ��K	����A�6*

logging/current_cost��;oi�C+       ��K	9���A�6*

logging/current_cost;�;T�^�+       ��K	�0��A�6*

logging/current_cost�݄;�~�+       ��K	�`��A�6*

logging/current_cost)؄;��t.+       ��K	����A�6*

logging/current_cost�҄;\�m�+       ��K	����A�6*

logging/current_cost�̈́;�z>t+       ��K	����A�6*

logging/current_costoʄ;�ySu+       ��K	�'��A�6*

logging/current_cost�ń;��k+       ��K	XU��A�6*

logging/current_cost���;�1d+       ��K	u���A�6*

logging/current_cost���;d�R�+       ��K	?���A�7*

logging/current_cost���;72dQ+       ��K	����A�7*

logging/current_costj��;_�+       ��K	���A�7*

logging/current_cost,��;hql�+       ��K	U<��A�7*

logging/current_costc��;e�Y�+       ��K	Dk��A�7*

logging/current_cost���;���+       ��K	.���A�7*

logging/current_cost뢄;��	�+       ��K	f���A�7*

logging/current_cost蜄;�m�+       ��K	���A�7*

logging/current_costY��;2U$R+       ��K	�<��A�7*

logging/current_cost=|�;.�+       ��K	�o��A�7*

logging/current_cost�k�;��`+       ��K	����A�7*

logging/current_cost a�;���b+       ��K	���A�7*

logging/current_cost�R�;�
Hy+       ��K	w���A�7*

logging/current_costM�;�ڼ�+       ��K	r*��A�7*

logging/current_costYG�;��o�+       ��K	�Y��A�7*

logging/current_costw@�;>]�+       ��K	����A�7*

logging/current_cost�9�;͡��+       ��K		���A�7*

logging/current_cost�2�;z��?+       ��K	a���A�7*

logging/current_cost�+�;�4��+       ��K	�$��A�7*

logging/current_cost-%�;L=9+       ��K	FW��A�7*

logging/current_cost� �;���+       ��K	Ć��A�7*

logging/current_costg�;��+       ��K	ǳ��A�7*

logging/current_cost��;<r3+       ��K	(���A�7*

logging/current_cost��;[h�+       ��K	o��A�7*

logging/current_cost��;�+       ��K	:B��A�7*

logging/current_cost*�;J:+       ��K	n��A�7*

logging/current_cost�;��>;+       ��K	����A�8*

logging/current_cost�;�8+       ��K	����A�8*

logging/current_cost��;V��+       ��K	N���A�8*

logging/current_cost��;�B(\+       ��K	�)��A�8*

logging/current_cost1��;���+       ��K	XX��A�8*

logging/current_cost���;s�Y�+       ��K	S���A�8*

logging/current_cost���;a�
+       ��K	����A�8*

logging/current_costQ��;���+       ��K	����A�8*

logging/current_cost��;���+       ��K	3 ��A�8*

logging/current_cost��;���+       ��K	"E ��A�8*

logging/current_cost>�;�C�+       ��K	&r ��A�8*

logging/current_costa�;Ӓ6+       ��K	G� ��A�8*

logging/current_cost�;`�j�+       ��K	r� ��A�8*

logging/current_cost �;K_ES+       ��K	�!��A�8*

logging/current_cost��;�랼+       ��K	�1!��A�8*

logging/current_cost��;,�B�+       ��K	m_!��A�8*

logging/current_costH�;fż|+       ��K	܍!��A�8*

logging/current_cost�߃;�I +       ��K	�!��A�8*

logging/current_cost�݃;]Z�L+       ��K	��!��A�8*

logging/current_cost�܃;��@+       ��K	"��A�8*

logging/current_cost�ۃ;Ё`_+       ��K	GL"��A�8*

logging/current_cost�ڃ;oI�m+       ��K	�y"��A�8*

logging/current_cost^ك;��y�+       ��K	��"��A�8*

logging/current_cost�׃;�%j+       ��K	��"��A�8*

logging/current_cost�׃;�ӳA+       ��K	�#��A�8*

logging/current_costwփ;>*��+       ��K	�/#��A�8*

logging/current_costՃ;�J��+       ��K	3]#��A�9*

logging/current_cost�ԃ;Y��+       ��K	ъ#��A�9*

logging/current_cost3ԃ;��y�+       ��K	&�#��A�9*

logging/current_cost>Ӄ;q���+       ��K	3�#��A�9*

logging/current_costuӃ;vi	�+       ��K	$��A�9*

logging/current_cost҃;M�N�+       ��K	�B$��A�9*

logging/current_costIу;GNC+       ��K	�p$��A�9*

logging/current_cost'҃;�f�+       ��K	P�$��A�9*

logging/current_costLу;e�v+       ��K	��$��A�9*

logging/current_costу;���+       ��K	��$��A�9*

logging/current_costЃ;$T

+       ��K	 /%��A�9*

logging/current_cost/σ;�AH+       ��K	�]%��A�9*

logging/current_cost�σ;oڎ�+       ��K	7�%��A�9*

logging/current_cost(σ;ׁ�+       ��K	��%��A�9*

logging/current_cost�΃;gw�+       ��K	��%��A�9*

logging/current_costO΃;¯2)+       ��K	�&��A�9*

logging/current_cost�΃;x;}�+       ��K	3K&��A�9*

logging/current_cost6΃;�~.+       ��K	�y&��A�9*

logging/current_cost)΃;0�9+       ��K	X�&��A�9*

logging/current_cost�̓;떓�+       ��K	p�&��A�9*

logging/current_cost�̓;)�:�+       ��K	'��A�9*

logging/current_cost�̓;��iP+       ��K	+7'��A�9*

logging/current_cost<̓;+��+       ��K	�d'��A�9*

logging/current_cost�̃;&�m�+       ��K	��'��A�9*

logging/current_cost�̃;����+       ��K	��'��A�9*

logging/current_cost�̃;�@�+       ��K	a�'��A�:*

logging/current_costf̃;�҅0+       ��K	5-(��A�:*

logging/current_cost�˃;ei1�+       ��K	@`(��A�:*

logging/current_cost�̃;�A��+       ��K	��(��A�:*

logging/current_cost�˃;	mU+       ��K	-�(��A�:*

logging/current_cost�̃;u�ִ+       ��K	��(��A�:*

logging/current_cost�˃;���+       ��K	�)��A�:*

logging/current_cost�˃;�9a+       ��K	�G)��A�:*

logging/current_cost�˃;���C+       ��K	�t)��A�:*

logging/current_cost_˃;���U+       ��K	�)��A�:*

logging/current_cost�˃;�>z�+       ��K	��)��A�:*

logging/current_costM˃; �+       ��K	3*��A�:*

logging/current_costy˃;p -�+       ��K	�4*��A�:*

logging/current_costl˃;/Hp�+       ��K	cg*��A�:*

logging/current_costL˃;�� �+       ��K	�*��A�:*

logging/current_cost\˃;�p��+       ��K	4�*��A�:*

logging/current_cost�ʃ;ۣ +       ��K	��*��A�:*

logging/current_cost˃;�+       ��K	6&+��A�:*

logging/current_cost�ʃ;���+       ��K	]X+��A�:*

logging/current_cost˃;Yg�+       ��K	�+��A�:*

logging/current_costA˃;%��+       ��K	�+��A�:*

logging/current_cost�ʃ;�~�+       ��K	x�+��A�:*

logging/current_cost�ʃ;Zj�7+       ��K	�,��A�:*

logging/current_cost˃;r|�N+       ��K	sF,��A�:*

logging/current_cost�ʃ;7��+       ��K	�x,��A�:*

logging/current_cost�ʃ;��<%+       ��K	��,��A�:*

logging/current_cost�ʃ;R�+       ��K	��,��A�;*

logging/current_cost�ʃ;ms��+       ��K	@	-��A�;*

logging/current_cost�ʃ;���+       ��K	G8-��A�;*

logging/current_costʃ;n��`+       ��K	Zg-��A�;*

logging/current_cost�˃;�<ރ+       ��K	�-��A�;*

logging/current_cost̃;�+:f+       ��K	��-��A�;*

logging/current_cost�ʃ;�5[�+       ��K	��-��A�;*

logging/current_cost�Ƀ;�F�+       ��K	`#.��A�;*

logging/current_cost�Ƀ;4d��+       ��K	�Q.��A�;*

logging/current_cost�Ƀ;XU�	+       ��K	u.��A�;*

logging/current_cost]ʃ;��,�+       ��K	u�.��A�;*

logging/current_costEʃ;�l�+       ��K	s�.��A�;*

logging/current_costʃ;c~X�+       ��K	 /��A�;*

logging/current_cost�Ƀ;����+       ��K	�4/��A�;*

logging/current_cost�Ƀ;�F�2+       ��K	sb/��A�;*

logging/current_cost[Ƀ;�=~�+       ��K	�/��A�;*

logging/current_cost�ʃ;]�z>+       ��K	�/��A�;*

logging/current_cost)Ƀ;U-F+       ��K	��/��A�;*

logging/current_cost&Ƀ;8^3i+       ��K	� 0��A�;*

logging/current_costlɃ;7L��+       ��K	�P0��A�;*

logging/current_cost#Ƀ;����+       ��K	X}0��A�;*

logging/current_costʃ;1Pt�+       ��K	��0��A�;*

logging/current_cost
Ƀ;S�6+       ��K	��0��A�;*

logging/current_cost0Ƀ;pF+       ��K	�1��A�;*

logging/current_costMɃ;$11?+       ��K	41��A�;*

logging/current_cost�ȃ;��!�+       ��K	�`1��A�<*

logging/current_costɃ;���m+       ��K	ۏ1��A�<*

logging/current_cost�Ƀ;��7+       ��K	6�1��A�<*

logging/current_cost�ȃ;"��+       ��K	��1��A�<*

logging/current_cost�Ƀ;��=�+       ��K	�2��A�<*

logging/current_cost0Ƀ;(.��+       ��K	�I2��A�<*

logging/current_cost�ȃ;�+       ��K	�w2��A�<*

logging/current_costVȃ;��=�+       ��K	(�2��A�<*

logging/current_cost�Ƀ;P��5+       ��K	5�2��A�<*

logging/current_cost�ƃ;����+       ��K	i�2��A�<*

logging/current_cost+ƃ;�b`�+       ��K	[,3��A�<*

logging/current_costAƃ;�l+       ��K	�Z3��A�<*

logging/current_cost�Ń;��&!+       ��K	��3��A�<*

logging/current_cost�ƃ;
>
�+       ��K	�3��A�<*

logging/current_costzƃ;:»,+       ��K	�3��A�<*

logging/current_costMƃ;!u��+       ��K	�4��A�<*

logging/current_costWă;$��+       ��K	�C4��A�<*

logging/current_cost.Ń;x���+       ��K	�q4��A�<*

logging/current_cost�Ń;(1w9+       ��K	=�4��A�<*

logging/current_cost�Ń;f��+       ��K	��4��A�<*

logging/current_costjă;pI�+       ��K	��4��A�<*

logging/current_costŃ;�H2(+       ��K	�25��A�<*

logging/current_costEŃ;3O�q+       ��K	Ab5��A�<*

logging/current_cost�ă;�w�+       ��K	d�5��A�<*

logging/current_cost�ă; E�D+       ��K	�5��A�<*

logging/current_costKƃ;��+       ��K	��5��A�<*

logging/current_cost�Ń;�o+       ��K	/6��A�=*

logging/current_cost=ă;���%+       ��K	@L6��A�=*

logging/current_cost�Ã;����+       ��K	�z6��A�=*

logging/current_cost�Ã;����+       ��K	_�6��A�=*

logging/current_cost$ă;`�+       ��K	��6��A�=*

logging/current_cost�Ã;�φ�+       ��K	
7��A�=*

logging/current_cost�Ã;ɇ�+       ��K	.97��A�=*

logging/current_cost�Ã;TeC+       ��K	�i7��A�=*

logging/current_costSÃ;�w��+       ��K	>�7��A�=*

logging/current_cost�Ã;�ꒀ+       ��K	��7��A�=*

logging/current_cost�Ã;��9�+       ��K	#�7��A�=*

logging/current_costă;�1��+       ��K	�)8��A�=*

logging/current_cost�Ã;0� �+       ��K	�W8��A�=*

logging/current_cost�ă;% f+       ��K	�8��A�=*

logging/current_costQă;l�3+       ��K	�8��A�=*

logging/current_costbÃ;��*/+       ��K	r�8��A�=*

logging/current_cost5Ã;4�9�+       ��K	�9��A�=*

logging/current_costeÃ;��+       ��K	E9��A�=*

logging/current_cost�Ã;��+       ��K	�z9��A�=*

logging/current_cost�Ã;`kH+       ��K	�9��A�=*

logging/current_cost.Ń;Z��D+       ��K	�9��A�=*

logging/current_costgă;
�tR+       ��K	 :��A�=*

logging/current_cost�Ã;e�+       ��K	B:��A�=*

logging/current_cost&ă;�T7�+       ��K	�p:��A�=*

logging/current_cost'ă;
�;+       ��K	Ӝ:��A�=*

logging/current_cost�Ã;��+       ��K	5�:��A�=*

logging/current_cost6ă;�#�+       ��K	��:��A�>*

logging/current_costgŃ;8tv�+       ��K	�';��A�>*

logging/current_cost�ă;�yv+       ��K	�T;��A�>*

logging/current_cost�;~K��+       ��K	�;��A�>*

logging/current_cost�;�gx�+       ��K	�;��A�>*

logging/current_cost�;����+       ��K	�2<��A�>*

logging/current_cost8;���+       ��K	>�<��A�>*

logging/current_cost3Ã;�B��+       ��K	��<��A�>*

logging/current_cost�ă;�ھ*+       ��K	�=��A�>*

logging/current_costNÃ;g;*p+       ��K	�X=��A�>*

logging/current_cost�;�2�+       ��K	-�=��A�>*

logging/current_cost�;r�y@+       ��K	�=��A�>*

logging/current_cost�;���+       ��K	��=��A�>*

logging/current_cost-;b��+       ��K	�0>��A�>*

logging/current_cost�;�1�^+       ��K	�n>��A�>*

logging/current_cost�;��+       ��K	��>��A�>*

logging/current_costj;�Eq�+       ��K	`�>��A�>*

logging/current_cost<Ã;��+       ��K	C?��A�>*

logging/current_cost�;��ʠ+       ��K	eP?��A�>*

logging/current_costH;���+       ��K	�?��A�>*

logging/current_cost���;@g�+       ��K	9�?��A�>*

logging/current_cost:Ã;��S+       ��K	��?��A�>*

logging/current_cost�Ã;�q��+       ��K	w@��A�>*

logging/current_cost���;���+       ��K	eO@��A�>*

logging/current_costyÃ;"���+       ��K	܀@��A�>*

logging/current_cost�Ã;�,/{+       ��K	��@��A�?*

logging/current_cost���;*��k+       ��K	I�@��A�?*

logging/current_cost;]�q+       ��K	IA��A�?*

logging/current_cost~;�;+       ��K	�DA��A�?*

logging/current_cost���;
��+       ��K	�rA��A�?*

logging/current_cost�;�P��+       ��K	��A��A�?*

logging/current_costfÃ;m�3�+       ��K	��A��A�?*

logging/current_cost?;S��+       ��K	��A��A�?*

logging/current_cost�;���+       ��K	�4B��A�?*

logging/current_cost���;`�d+       ��K	�iB��A�?*

logging/current_costY;0��+       ��K	ܚB��A�?*

logging/current_costl;p��+       ��K	N�B��A�?*

logging/current_costm;���E+       ��K	h�B��A�?*

logging/current_cost���;#�ǵ+       ��K	(C��A�?*

logging/current_cost;;���v+       ��K	fUC��A�?*

logging/current_cost}Ã;X.Q^+       ��K	;�C��A�?*

logging/current_cost-;=䱒+       ��K	��C��A�?*

logging/current_costp;Wo�+       ��K	��C��A�?*

logging/current_cost���;	}�Z+       ��K	�D��A�?*

logging/current_cost�;�Ma+       ��K	�MD��A�?*

logging/current_cost���;�mm�+       ��K	}D��A�?*

logging/current_costg��;w�1;+       ��K	��D��A�?*

logging/current_cost���;��xM+       ��K	�D��A�?*

logging/current_cost���;��5X+       ��K	�E��A�?*

logging/current_cost���;Bd& +       ��K	�<E��A�?*

logging/current_costu��;l�+       ��K	��E��A�?*

logging/current_costN��;)�*^+       ��K	N�E��A�@*

logging/current_cost���;�Ò�+       ��K	9F��A�@*

logging/current_cost�;��m+       ��K	V?F��A�@*

logging/current_cost���;i���+       ��K	F��A�@*

logging/current_cost;�$�+       ��K	��F��A�@*

logging/current_cost���;�
�+       ��K	MG��A�@*

logging/current_cost���;���9+       ��K	�UG��A�@*

logging/current_costn��;��6�+       ��K	c�G��A�@*

logging/current_cost-��;?k�+       ��K	��G��A�@*

logging/current_cost`��;����+       ��K	�H��A�@*

logging/current_cost���;&$�,+       ��K	WEH��A�@*

logging/current_cost���;[�1v+       ��K	B�H��A�@*

logging/current_cost���;�vSX+       ��K	��H��A�@*

logging/current_cost
;�9��+       ��K	�I��A�@*

logging/current_cost2;�Z�+       ��K	r?I��A�@*

logging/current_costÃ;s֡M+       ��K	�zI��A�@*

logging/current_coste;r�+       ��K	k�I��A�@*

logging/current_cost���;�Z�+       ��K	��I��A�@*

logging/current_cost���;G��+       ��K	�J��A�@*

logging/current_cost���;x��j+       ��K	�EJ��A�@*

logging/current_cost%��;����+       ��K	5uJ��A�@*

logging/current_cost+��;���+       ��K	v�J��A�@*

logging/current_costU��;��`�+       ��K	��J��A�@*

logging/current_cost���;cXL+       ��K	<K��A�@*

logging/current_cost���;��+       ��K	�0K��A�@*

logging/current_costg��;-�%+       ��K	AcK��A�A*

logging/current_cost!��;*zn+       ��K	ޑK��A�A*

logging/current_cost���;�g(�+       ��K	o�K��A�A*

logging/current_costr;��v+       ��K	[�K��A�A*

logging/current_costn;#H�)+       ��K	�)L��A�A*

logging/current_cost"��;�\�+       ��K	�XL��A�A*

logging/current_cost���;x��+       ��K	��L��A�A*

logging/current_cost���;�]+       ��K	��L��A�A*

logging/current_cost���;��@+       ��K	�L��A�A*

logging/current_cost���;�zgB+       ��K	}M��A�A*

logging/current_cost���;��4�+       ��K	?M��A�A*

logging/current_cost4��;~VEp+       ��K	1kM��A�A*

logging/current_cost���;&M9`+       ��K	S�M��A�A*

logging/current_cost!;�=�+       ��K	 �M��A�A*

logging/current_cost[;4k�L+       ��K	��M��A�A*

logging/current_cost;�C�+       ��K	6&N��A�A*

logging/current_costF;ݘ;+       ��K	%WN��A�A*

logging/current_cost<��;�ا�+       ��K	�N��A�A*

logging/current_costH��;L< �+       ��K	��N��A�A*

logging/current_costh��;L��+       ��K	)O��A�A*

logging/current_cost���;���x+       ��K	�YO��A�A*

logging/current_cost���;�R3+       ��K	��O��A�A*

logging/current_cost���;v�z�+       ��K	ֽO��A�A*

logging/current_cost���;��<8+       ��K	n�O��A�A*

logging/current_cost]��;;�- +       ��K	�P��A�A*

logging/current_cost���;v���+       ��K	�KP��A�A*

logging/current_costk��;���+       ��K	|P��A�B*

logging/current_cost~��;0���+       ��K	s�P��A�B*

logging/current_costs��;�F7�+       ��K	��P��A�B*

logging/current_cost%��;af +       ��K	DQ��A�B*

logging/current_cost���;s��+       ��K	�CQ��A�B*

logging/current_costB��;Ĝ��+       ��K	vQ��A�B*

logging/current_costZ;�x�+       ��K	�Q��A�B*

logging/current_cost_Ã;H���+       ��K	��Q��A�B*

logging/current_cost���;GT+       ��K	gR��A�B*

logging/current_cost���;?�\�+       ��K	h8R��A�B*

logging/current_cost��;x��+       ��K	|qR��A�B*

logging/current_cost;�`m�+       ��K	\�R��A�B*

logging/current_costo��;9�>+       ��K	��R��A�B*

logging/current_cost���;��|'+       ��K	dS��A�B*

logging/current_cost���;����+       ��K	�1S��A�B*

logging/current_costv��;��XS+       ��K	,dS��A�B*

logging/current_cost���;Ӟ�+       ��K	�S��A�B*

logging/current_cost���;eQ�s+       ��K	ۼS��A�B*

logging/current_cost��;��ƪ+       ��K	0�S��A�B*

logging/current_cost���;�q=+       ��K	)T��A�B*

logging/current_costE��;�A+       ��K	4XT��A�B*

logging/current_costA��;B֞ +       ��K	хT��A�B*

logging/current_cost޿�;H/U+       ��K	��T��A�B*

logging/current_cost˿�;�(ٷ+       ��K	��T��A�B*

logging/current_costͿ�;�2Y:+       ��K	�U��A�B*

logging/current_cost���;x-~+       ��K	9HU��A�B*

logging/current_cost5��;�z�+       ��K	�vU��A�C*

logging/current_cost���;�.�+       ��K	�U��A�C*

logging/current_cost��;Ew+       ��K	��U��A�C*

logging/current_costѿ�;@�+       ��K	�V��A�C*

logging/current_costd��;�E�0+       ��K	h/V��A�C*

logging/current_cost���;=��+       ��K	�\V��A�C*

logging/current_cost��;:/�+       ��K	�V��A�C*

logging/current_cost/��;�e++       ��K	�V��A�C*

logging/current_cost���;*�\F+       ��K	��V��A�C*

logging/current_costf��;zu��+       ��K	�W��A�C*

logging/current_cost���;�I�/+       ��K		>W��A�C*

logging/current_cost���;GY#�+       ��K	�jW��A�C*

logging/current_cost���;&v�d+       ��K	>�W��A�C*

logging/current_cost���;NL�$+       ��K	��W��A�C*

logging/current_cost@��;�J+       ��K	��W��A�C*

logging/current_cost���;F�fM+       ��K	�)X��A�C*

logging/current_cost���;�UHc+       ��K	�XX��A�C*

logging/current_cost���;���+       ��K	;�X��A�C*

logging/current_cost���;?�)�+       ��K	�X��A�C*

logging/current_cost���;��r�+       ��K	'�X��A�C*

logging/current_costh��;�JoV+       ��K	�Y��A�C*

logging/current_costs��;T�·+       ��K	�@Y��A�C*

logging/current_costg��;Ϥ��+       ��K	poY��A�C*

logging/current_costY��;��!�+       ��K	�Y��A�C*

logging/current_cost{��;:��+       ��K	U�Y��A�C*

logging/current_cost���;���+       ��K	S�Y��A�D*

logging/current_cost���;`�*�+       ��K	�#Z��A�D*

logging/current_cost���;�R;+       ��K	�RZ��A�D*

logging/current_cost���;#YM+       ��K	5�Z��A�D*

logging/current_costE��;�s,+       ��K	R�Z��A�D*

logging/current_cost��;�>�+       ��K	c�Z��A�D*

logging/current_costk��;]�+       ��K	T[��A�D*

logging/current_cost���;�~��+       ��K	G>[��A�D*

logging/current_costQ��;�X�+       ��K	�j[��A�D*

logging/current_cost��;	��+       ��K	�[��A�D*

logging/current_costv��;�
��+       ��K	��[��A�D*

logging/current_cost��;Fʗ+       ��K	��[��A�D*

logging/current_costK��;>�L�+       ��K	,\��A�D*

logging/current_cost��;���y+       ��K	HZ\��A�D*

logging/current_cost���;&И�+       ��K	�\��A�D*

logging/current_cost���;�G~�+       ��K	H�\��A�D*

logging/current_cost���;kX8�+       ��K	��\��A�D*

logging/current_cost��;h��=+       ��K	m]��A�D*

logging/current_cost���;|���+       ��K	�M]��A�D*

logging/current_cost���;�
�+       ��K	}]��A�D*

logging/current_cost���;a��~+       ��K	��]��A�D*

logging/current_cost���;F�`+       ��K	��]��A�D*

logging/current_cost1��;w��+       ��K	:^��A�D*

logging/current_cost���;��+�+       ��K	�8^��A�D*

logging/current_cost7��;��
.+       ��K	/f^��A�D*

logging/current_cost���;����+       ��K	��^��A�D*

logging/current_cost޿�;N��+       ��K	8�^��A�E*

logging/current_cost�;*\�+       ��K	Z�^��A�E*

logging/current_costJ��;�+,+       ��K	~"_��A�E*

logging/current_cost��;5�ߎ+       ��K	[O_��A�E*

logging/current_cost���;y
,+       ��K	�_��A�E*

logging/current_costj��;��'�+       ��K	Y�_��A�E*

logging/current_cost���;	���+       ��K	X�_��A�E*

logging/current_cost��;�oO+       ��K	3`��A�E*

logging/current_cost���;���+       ��K	�<`��A�E*

logging/current_cost��;���+       ��K	#k`��A�E*

logging/current_cost%��;2�Wa+       ��K	Ǘ`��A�E*

logging/current_cost��;y�0�+       ��K	<�`��A�E*

logging/current_cost���;n�";+       ��K	I�`��A�E*

logging/current_cost|��;����+       ��K	!&a��A�E*

logging/current_cost_��;�3�+       ��K	oSa��A�E*

logging/current_cost\��;ū�>+       ��K	��a��A�E*

logging/current_cost���;�
�+       ��K	�a��A�E*

logging/current_cost0��;7CV�+       ��K	�a��A�E*

logging/current_cost���;U�Z�+       ��K	b��A�E*

logging/current_cost���;n�	C+       ��K	Cb��A�E*

logging/current_cost���;M�+       ��K	�rb��A�E*

logging/current_cost较;��q+       ��K	˟b��A�E*

logging/current_cost���;�䀽+       ��K	C�b��A�E*

logging/current_costj��;[���+       ��K	$ c��A�E*

logging/current_cost��;� �+       ��K	 .c��A�E*

logging/current_costz��;��U+       ��K	*Zc��A�F*

logging/current_cost��;X.
+       ��K	�c��A�F*

logging/current_costп�;+ͺ*+       ��K	ڴc��A�F*

logging/current_costl��;�F�+       ��K	��c��A�F*

logging/current_costD��;���3+       ��K	d��A�F*

logging/current_cost9��;
<�L+       ��K	,>d��A�F*

logging/current_cost��;�8X+       ��K	lmd��A�F*

logging/current_cost���;���+       ��K	��d��A�F*

logging/current_cost3��;�]��+       ��K	��d��A�F*

logging/current_costj��;Ƕ�P+       ��K	��d��A�F*

logging/current_cost'��;q�5+       ��K	�&e��A�F*

logging/current_cost��;�7��+       ��K	�Ue��A�F*

logging/current_cost心; 4�R+       ��K	6�e��A�F*

logging/current_cost���;mf��+       ��K	)�e��A�F*

logging/current_costϿ�;�RI +       ��K	I�e��A�F*

logging/current_cost��;�셸+       ��K	.f��A�F*

logging/current_cost��;k��0+       ��K	/?f��A�F*

logging/current_cost���;�G�+       ��K	vnf��A�F*

logging/current_cost
��;B��+       ��K	��f��A�F*

logging/current_cost���;Ƈ�r+       ��K	"�f��A�F*

logging/current_costg��;�{
�+       ��K	��f��A�F*

logging/current_cost���;�M/+       ��K	�+g��A�F*

logging/current_cost��;��B�+       ��K	[g��A�F*

logging/current_cost���;���+       ��K	_�g��A�F*

logging/current_cost)��;ќ�+       ��K	ȶg��A�F*

logging/current_cost���;�wsD+       ��K	M�g��A�F*

logging/current_costÿ�;$=K+       ��K	�h��A�G*

logging/current_cost���;Q=Š+       ��K	�Eh��A�G*

logging/current_cost�;���+       ��K	Vrh��A�G*

logging/current_cost���;�p�+       ��K	Ϣh��A�G*

logging/current_cost��;y��D+       ��K	w�h��A�G*

logging/current_cost���;��l+       ��K	�i��A�G*

logging/current_cost���;G�T�+       ��K	
1i��A�G*

logging/current_cost���;j3�+       ��K	Oai��A�G*

logging/current_cost��;��%+       ��K	�i��A�G*

logging/current_coste;#
1#+       ��K	!�i��A�G*

logging/current_cost^��;�HN�+       ��K	��i��A�G*

logging/current_cost6Ã;up3�+       ��K	�!j��A�G*

logging/current_cost��;���+       ��K	�Vj��A�G*

logging/current_cost���;���&+       ��K	4�j��A�G*

logging/current_costH��;�D��+       ��K	'�j��A�G*

logging/current_costܿ�;~~�>+       ��K	��j��A�G*

logging/current_cost;��;�إ�+       ��K	�k��A�G*

logging/current_costJ��;ñA�+       ��K	Ak��A�G*

logging/current_cost龃;��q�+       ��K	�ok��A�G*

logging/current_cost^��;7�x�+       ��K	�k��A�G*

logging/current_cost!��;Y�5+       ��K	��k��A�G*

logging/current_cost��;���|+       ��K	g l��A�G*

logging/current_cost��;�&E+       ��K	�-l��A�G*

logging/current_costb��;0�`~+       ��K	 al��A�G*

logging/current_cost���;����+       ��K	R�l��A�G*

logging/current_cost���;&HP�+       ��K	h�l��A�G*

logging/current_costӿ�;��+       ��K	g�l��A�H*

logging/current_cost ��;;���+       ��K	Jm��A�H*

logging/current_cost��;���B+       ��K	�Pm��A�H*

logging/current_cost ��;�
�+       ��K	�m��A�H*

logging/current_costY��;�;�+       ��K	R�m��A�H*

logging/current_cost���;����+       ��K	��m��A�H*

logging/current_cost]��;��9�+       ��K	�n��A�H*

logging/current_costq��;���+       ��K	*>n��A�H*

logging/current_costO��;�+       ��K	�kn��A�H*

logging/current_cost��;oR�!+       ��K	Z�n��A�H*

logging/current_cost���;���R+       ��K	
�n��A�H*

logging/current_cost���;2^I�+       ��K	o��A�H*

logging/current_costZ��;�w�+       ��K	�0o��A�H*

logging/current_costF��;zh6+       ��K	#\o��A�H*

logging/current_costI��;)�5+       ��K	+�o��A�H*

logging/current_cost���;/+       ��K	ںo��A�H*

logging/current_costk��;�?�+       ��K	X�o��A�H*

logging/current_cost��;��6+       ��K	p��A�H*

logging/current_costϿ�;1��+       ��K	�Ep��A�H*

logging/current_cost|��;!��+       ��K	sp��A�H*

logging/current_costz��;��+       ��K	��p��A�H*

logging/current_cost���;�°+       ��K	��p��A�H*

logging/current_cost;�;����+       ��K	��p��A�H*

logging/current_cost���;{�km+       ��K	(-q��A�H*

logging/current_cost��;_)�+       ��K	�Zq��A�H*

logging/current_cost*��;gbE�+       ��K	g�q��A�I*

logging/current_cost<��;���+       ��K	z�q��A�I*

logging/current_cost8��;L�+       ��K	��q��A�I*

logging/current_costl��;�Lo|+       ��K	r��A�I*

logging/current_cost0��;~u
8+       ��K	G@r��A�I*

logging/current_costA��;���t+       ��K	nr��A�I*

logging/current_cost���;�m�e+       ��K	��r��A�I*

logging/current_cost%��;n^K0+       ��K	�r��A�I*

logging/current_cost���;���	+       ��K	��r��A�I*

logging/current_cost~��;\3�+       ��K	L#s��A�I*

logging/current_cost龃;F��+       ��K	#Qs��A�I*

logging/current_costQ��;k��W+       ��K	�s��A�I*

logging/current_cost쾃;��*h+       ��K	��s��A�I*

logging/current_cost)��;h��+       ��K	��s��A�I*

logging/current_cost���;F�h+       ��K	7
t��A�I*

logging/current_cost ��;%Y��+       ��K	�7t��A�I*

logging/current_cost龃;�H��+       ��K	ft��A�I*

logging/current_cost��;�ڴA+       ��K	B�t��A�I*

logging/current_cost���;�b'+       ��K	`�t��A�I*

logging/current_costY��;�#�z+       ��K	E�t��A�I*

logging/current_cost���;s��t+       ��K	!u��A�I*

logging/current_costh��;�Ѥ�+       ��K	tNu��A�I*

logging/current_cost?��;�l��+       ��K	V}u��A�I*

logging/current_cost���;yoOG+       ��K	��u��A�I*

logging/current_cost���;��+       ��K	��u��A�I*

logging/current_cost���;��+       ��K	P
v��A�I*

logging/current_cost%��;��+       ��K	H6v��A�J*

logging/current_cost��;�Y�|+       ��K	�dv��A�J*

logging/current_cost1��;x�g�+       ��K	2�v��A�J*

logging/current_cost��;G�*G+       ��K	��v��A�J*

logging/current_cost)��;�T�+       ��K	w�v��A�J*

logging/current_cost���;����+       ��K	�w��A�J*

logging/current_cost���;˷�+       ��K	[Iw��A�J*

logging/current_cost5��;�A6+       ��K	^yw��A�J*

logging/current_costd��;*q�+       ��K	{�w��A�J*

logging/current_costM��;�Bv+       ��K	>�w��A�J*

logging/current_cost຃;^?��+       ��K	�x��A�J*

logging/current_cost(��;����+       ��K	g7x��A�J*

logging/current_cost~��;�I��+       ��K	�ex��A�J*

logging/current_cost��;��'+       ��K	&�x��A�J*

logging/current_cost���;6:'�+       ��K	%�x��A�J*

logging/current_cost��;�;��+       ��K	��x��A�J*

logging/current_costv��;ye�!+       ��K	$y��A�J*

logging/current_cost���;���+       ��K	:Qy��A�J*

logging/current_cost���;5:��+       ��K	�y��A�J*

logging/current_cost��;)�+       ��K	��y��A�J*

logging/current_cost�;�o݈+       ��K	��y��A�J*

logging/current_cost˷�;�)�@+       ��K	�z��A�J*

logging/current_costʷ�;�7�+       ��K	�@z��A�J*

logging/current_cost���;���+       ��K	�oz��A�J*

logging/current_cost췃;	�+       ��K	&�z��A�J*

logging/current_cost ��;�+       ��K	��z��A�K*

logging/current_cost䷃;8t�J+       ��K	��z��A�K*

logging/current_cost���;1A[m+       ��K	�){��A�K*

logging/current_cost_��;��+       ��K	Y{��A�K*

logging/current_cost��;�?o+       ��K	n�{��A�K*

logging/current_cost��;�۟+       ��K	R�{��A�K*

logging/current_cost.��;�h�
+       ��K	{!|��A�K*

logging/current_costB��;�%a�+       ��K	�^|��A�K*

logging/current_costֶ�;w/V\+       ��K	�|��A�K*

logging/current_cost���;*�L+       ��K	j�|��A�K*

logging/current_costC��;�X��+       ��K	}��A�K*

logging/current_cost���;(p)	+       ��K	5}��A�K*

logging/current_costG��;̇a+       ��K	�i}��A�K*

logging/current_cost���;E�w5+       ��K	S�}��A�K*

logging/current_cost]��;�!�+       ��K	�}��A�K*

logging/current_cost���;�,Q�+       ��K	�~��A�K*

logging/current_cost[��;T2�f+       ��K	�H~��A�K*

logging/current_cost1��;ZV �+       ��K	�|~��A�K*

logging/current_costM��;�h�p+       ��K	_�~��A�K*

logging/current_costk��;��=�+       ��K	w�~��A�K*

logging/current_cost۵�;�?i�+       ��K	��A�K*

logging/current_costm��;U��+       ��K	�>��A�K*

logging/current_coste��;5O�+       ��K	�o��A�K*

logging/current_cost���;h-e�+       ��K	\���A�K*

logging/current_cost굃;dp�+       ��K	O���A�K*

logging/current_costɴ�;���+       ��K	} ���A�K*

logging/current_costV��;��7�+       ��K	�.���A�L*

logging/current_cost���;HZ�+       ��K	aZ���A�L*

logging/current_costh��;aq�+       ��K	R����A�L*

logging/current_cost���;���+       ��K	۶���A�L*

logging/current_cost'��;E�?c+       ��K	&瀩�A�L*

logging/current_cost��;0���+       ��K	L���A�L*

logging/current_cost᳃;?i��+       ��K	RF���A�L*

logging/current_cost̴�;��S�+       ��K	ks���A�L*

logging/current_costﳃ;��u+       ��K	�����A�L*

logging/current_cost8��;Ia�+       ��K	�ԁ��A�L*

logging/current_cost���;%�F�+       ��K	����A�L*

logging/current_cost˲�;��O%+       ��K	�4���A�L*

logging/current_costi��;����+       ��K	l���A�L*

logging/current_costᲃ;l#�+       ��K	�����A�L*

logging/current_cost���;F+       ��K	7˂��A�L*

logging/current_cost���;�2ʹ+       ��K	�����A�L*

logging/current_cost���;����+       ��K	�,���A�L*

logging/current_costǲ�;����+       ��K	G\���A�L*

logging/current_cost��;k��+       ��K	B����A�L*

logging/current_cost4��;!�E+       ��K	Խ���A�L*

logging/current_cost���;�M��+       ��K	��A�L*

logging/current_cost��;�=+       ��K		!���A�L*

logging/current_cost���;���+       ��K	�U���A�L*

logging/current_cost���;h�q�+       ��K	����A�L*

logging/current_costj��;�Mz+       ��K	�����A�L*

logging/current_cost��;�г�+       ��K	�℩�A�L*

logging/current_cost���;���+       ��K	����A�M*

logging/current_cost벃;��F+       ��K	�]���A�M*

logging/current_costI��;����+       ��K	�����A�M*

logging/current_cost���;UG�+       ��K	K����A�M*

logging/current_cost۱�;��
+       ��K	�녩�A�M*

logging/current_cost���;{(C�+       ��K	����A�M*

logging/current_cost��;���+       ��K	�L���A�M*

logging/current_cost2��;2?w+       ��K	�|���A�M*

logging/current_costw��;���+       ��K	p����A�M*

logging/current_costձ�;_�Z"+       ��K	vچ��A�M*

logging/current_cost���;t�+       ��K	7���A�M*

logging/current_cost��;�)=M+       ��K	�=���A�M*

logging/current_costǰ�;|>+       ��K	�q���A�M*

logging/current_cost걃;��+       ��K	\����A�M*

logging/current_cost���;	��+       ��K	�χ��A�M*

logging/current_cost���;�i �+       ��K	�����A�M*

logging/current_cost]��;m��1+       ��K	O/���A�M*

logging/current_cost��;?j��+       ��K	�`���A�M*

logging/current_cost찃;��?+       ��K	�����A�M*

logging/current_costN��;���H+       ��K	�ǈ��A�M*

logging/current_cost+��;nӈ+       ��K	�����A�M*

logging/current_cost쯃;��`+       ��K	$���A�M*

logging/current_cost:��;�H�+       ��K	�X���A�M*

logging/current_cost=��;���+       ��K	 ����A�M*

logging/current_cost���;���+       ��K	����A�M*

logging/current_cost���;�<l+       ��K	
艩�A�N*

logging/current_cost-��;���+       ��K	[���A�N*

logging/current_cost���;~v7�+       ��K	(F���A�N*

logging/current_cost=��;4�